"""Simple PDF viewer for PyQt5.

Pages are rendered to images and shown in a QScrollArea. Rendering engines:
  1. PyMuPDF (`pip install pymupdf`)            -> recommended (fast, no external process)
  2. `pdftoppm` (poppler-utils package)         -> fallback when PyMuPDF is not installed
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import fitz  # PyMuPDF (old module name)
except Exception:  # pragma: no cover - depends on the environment
    try:
        import pymupdf as fitz  # new module name in PyMuPDF >= 1.24
    except Exception:
        fitz = None


def renderer_name():
    if fitz is not None:
        return "PyMuPDF"
    if shutil.which("pdftoppm"):
        return "pdftoppm"
    return None


def render_pdf(path, zoom=1.5):
    """Render all PDF pages -> list[QImage]. zoom 1.0 = 72 dpi."""
    path = str(path)
    if fitz is not None:
        images = []
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                img = QtGui.QImage(pix.samples, pix.width, pix.height, pix.stride, QtGui.QImage.Format_RGB888)
                images.append(img.copy())  # copy: the pix buffer is released when the loop moves on
        return images
    exe = shutil.which("pdftoppm")
    if not exe:
        raise RuntimeError("No PDF rendering engine available. Install PyMuPDF (pip install pymupdf) "
                           "or poppler-utils (sudo apt install poppler-utils).")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([exe, "-r", str(int(72 * zoom)), "-png", path, str(Path(tmp) / "p")], check=True, timeout=120)
        return [QtGui.QImage(str(f)) for f in sorted(Path(tmp).glob("p*.png"))]


class PdfViewer(QtWidgets.QScrollArea):
    """Shows a PDF as vertically stacked pages with zoom and page navigation."""

    pageChanged = QtCore.pyqtSignal(int, int)  # current page (1-based), total

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.setStyleSheet("QScrollArea { background: #5A6270; border: 1px solid #12263F; }")
        self._holder = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout(self._holder)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(10)
        self._layout.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.setWidget(self._holder)
        self._labels = []
        self._path = None
        self._zoom = None          # None = fit width
        self._base_images = []
        self.verticalScrollBar().valueChanged.connect(self._emit_page)
        self.show_message("No report yet.\nRun a classification and click “Detail Report”,\n"
                          "or select a row in the history list.")

    # ------------------------------------------------------------- API
    @property
    def path(self):
        return self._path

    def page_count(self):
        return len(self._labels)

    def show_message(self, text):
        self._clear()
        self._path = None
        lab = QtWidgets.QLabel(text)
        lab.setAlignment(QtCore.Qt.AlignCenter)
        lab.setStyleSheet("color: #E9ECF1; font-size: 13px; padding: 40px;")
        self._layout.addWidget(lab)
        self._labels = []

    def load(self, path):
        """Load and display the PDF. Raises an exception if rendering fails."""
        self._path = str(path)
        self._base_images = render_pdf(path, zoom=2.0)   # render once at ~144 dpi, scale on the Qt side
        self._rebuild()
        self.verticalScrollBar().setValue(0)

    def set_zoom(self, factor):
        """factor relative to fit-width = 1.0; None -> fit width."""
        self._zoom = factor
        self._rebuild()

    def zoom_by(self, ratio):
        cur = self._zoom or 1.0
        self.set_zoom(max(0.4, min(3.0, cur * ratio)))

    def fit_width(self):
        self.set_zoom(None)

    def goto_page(self, n):
        if 1 <= n <= len(self._labels):
            self.verticalScrollBar().setValue(self._labels[n - 1].y() + self._holder.layout().contentsMargins().top()
                                              - 6)

    def current_page(self):
        y = self.verticalScrollBar().value() + self.viewport().height() // 3
        cur = 1
        for i, lab in enumerate(self._labels, 1):
            if lab.y() <= y:
                cur = i
        return cur

    # ------------------------------------------------------------- internal
    def _clear(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._labels = []

    def _rebuild(self):
        self._clear()
        if not self._base_images:
            return
        avail = max(200, self.viewport().width() - 40)
        for img in self._base_images:
            target_w = avail * (self._zoom or 1.0)
            pm = QtGui.QPixmap.fromImage(img).scaledToWidth(int(target_w), QtCore.Qt.SmoothTransformation)
            lab = QtWidgets.QLabel()
            lab.setPixmap(pm)
            lab.setFixedSize(pm.size())
            lab.setStyleSheet("background: white; border: 1px solid #222;")
            self._layout.addWidget(lab, 0, QtCore.Qt.AlignHCenter)
            self._labels.append(lab)
        QtCore.QTimer.singleShot(40, self._emit_page)     # after the layout has placed the pages

    def _emit_page(self, *_):
        if self._labels:
            self.pageChanged.emit(self.current_page(), len(self._labels))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._base_images and self._zoom is None:
            QtCore.QTimer.singleShot(0, self._rebuild)


class PdfToolbar(QtWidgets.QWidget):
    """Navigation/zoom buttons for a PdfViewer."""

    openExternal = QtCore.pyqtSignal()
    regenerate = QtCore.pyqtSignal()

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        def btn(text, tip, slot):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.setMinimumHeight(28)
            b.clicked.connect(slot)
            return b

        self.btn_prev = btn("◀", "Previous page", lambda: viewer.goto_page(viewer.current_page() - 1))
        self.btn_next = btn("▶", "Next page", lambda: viewer.goto_page(viewer.current_page() + 1))
        self.lbl = QtWidgets.QLabel("– / –")
        self.lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl.setMinimumWidth(60)
        self.btn_out = btn("−", "Zoom out", lambda: viewer.zoom_by(1 / 1.2))
        self.btn_in = btn("+", "Zoom in", lambda: viewer.zoom_by(1.2))
        self.btn_fit = btn("Fit", "Fit to width", viewer.fit_width)
        self.btn_ext = btn("Open ↗", "Open the PDF with the system default application", self.openExternal.emit)
        self.btn_regen = btn("Regenerate", "Regenerate the PDF report", self.regenerate.emit)
        for i, w in enumerate((self.btn_prev, self.lbl, self.btn_next, self.btn_out, self.btn_in, self.btn_fit)):
            lay.addWidget(w, 0, i)
        lay.addWidget(self.btn_ext, 1, 0, 1, 3)
        lay.addWidget(self.btn_regen, 1, 3, 1, 3)
        viewer.pageChanged.connect(lambda cur, tot: self.lbl.setText(f"{cur} / {tot}"))

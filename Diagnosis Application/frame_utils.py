"""Qt display utilities: OpenCV frame -> QPixmap conversion, virtual Y-axis overlay, ROI preview, message boxes."""
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


def bgr_to_pixmap(bgr):
    """BGR ndarray -> QPixmap (the data are copied so it stays valid after the array is released)."""
    h, w, ch = bgr.shape
    img = QtGui.QImage(bgr.data, w, h, ch * w, QtGui.QImage.Format_BGR888).copy()
    return QtGui.QPixmap.fromImage(img)


def make_scene(view, smooth=True):
    """Attach a new QGraphicsScene to the QGraphicsView and return it."""
    scene = QtWidgets.QGraphicsScene()
    view.setScene(scene)
    view.setRenderHint(QtGui.QPainter.Antialiasing)
    if smooth:
        view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
    return scene


def draw_axis_overlay(pixmap):
    """Draw the virtual Y axis (green line) and the Albumin/Creatinine labels on the preview pixmap."""
    p = QtGui.QPainter(pixmap)
    cx = pixmap.width() // 2
    p.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 2))
    p.drawLine(cx, 0, cx, pixmap.height())
    font = QtGui.QFont()
    font.setPointSize(12)
    font.setBold(True)
    p.setFont(font)
    fm = p.fontMetrics()
    p.setPen(QtGui.QColor(0, 255, 0))
    p.drawText(cx // 2 - fm.boundingRect("Albumin").width() // 2, 30, "Albumin")
    p.setPen(QtGui.QColor(255, 0, 0))
    p.drawText(cx + cx // 2 - fm.boundingRect("Creatinine").width() // 2, 30, "Creatinine")
    p.end()
    return pixmap


def show_live_frame(view, scene, frame, size):
    """Show the camera frame in `view` (target size `size` = (w, h)) with the Y-axis overlay."""
    h, w = frame.shape[:2]
    s = min(size[0] / w, size[1] / h)
    small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))))
    pm = draw_axis_overlay(bgr_to_pixmap(small))
    scene.clear()
    scene.addPixmap(pm)
    view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


def show_roi_preview(view, scene, bgr, size=221):
    """Preview of the cropped ROI in a small QGraphicsView."""
    if bgr is None:
        scene.clear()
        return
    scene.clear()
    scene.addPixmap(bgr_to_pixmap(cv2.resize(bgr, (size, size))))
    view.fitInView(scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)


def info(parent, title, text):
    QtWidgets.QMessageBox.information(parent, title, text)


def warn(parent, title, text):
    QtWidgets.QMessageBox.warning(parent, title, text)


def ask(parent, title, text):
    """Yes/No dialog (default No). Returns True for Yes."""
    box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, title, text,
                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, parent)
    box.setDefaultButton(QtWidgets.QMessageBox.No)
    return box.exec_() == QtWidgets.QMessageBox.Yes

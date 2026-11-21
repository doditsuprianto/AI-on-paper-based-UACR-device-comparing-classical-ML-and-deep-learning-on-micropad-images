#!/usr/bin/env bash
# Install the "Intelligent POCT Diagnostics" shortcut (Desktop icon + application menu).
#
#   bash install_shortcut.sh               install the Desktop + menu shortcut
#   bash install_shortcut.sh --autostart   ... and start the application automatically at desktop login
#   bash install_shortcut.sh --remove      remove all shortcuts (the application itself is not removed)
#
# Safe to run repeatedly. This script is self-contained: it creates run_poct.sh (launcher) and the application icon
# if they do not exist yet, then writes the .desktop files with the path of this folder.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
APP_NAME="Intelligent POCT Diagnostics"
LAUNCHER="$APP_DIR/run_poct.sh"
ICON="$APP_DIR/logo/poct_icon.png"
MENU_DIR="$HOME/.local/share/applications"
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
[ -n "$DESKTOP_DIR" ] && [ -d "$DESKTOP_DIR" ] || DESKTOP_DIR="$HOME/Desktop"

if [ "${1:-}" = "--remove" ]; then
    rm -f "$MENU_DIR/poct.desktop" "$AUTOSTART_DIR/poct.desktop" "$DESKTOP_DIR/poct.desktop"
    command -v update-desktop-database >/dev/null && update-desktop-database "$MENU_DIR" 2>/dev/null || true
    echo "Shortcuts removed (menu, Desktop, autostart). The application folder was not touched."
    exit 0
fi

[ -f "$APP_DIR/DiagnosticsV2.py" ] || { echo "ERROR: DiagnosticsV2.py not found in $APP_DIR"; exit 1; }
[ -x "$APP_DIR/.venv/bin/python" ] || echo "WARNING: $APP_DIR/.venv does not exist; the launcher will use the system python3."

# ------------------------------------------------------------------ 1. launcher
cat > "$LAUNCHER" <<'EOF'
#!/usr/bin/env bash
# Launcher of Intelligent POCT Diagnostics. Called by the shortcut; it can also be run manually: ./run_poct.sh
APP_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$APP_DIR" || exit 1

export DISPLAY="${DISPLAY:-:0}"
if [ -z "${XAUTHORITY:-}" ] && [ -f "$HOME/.Xauthority" ]; then export XAUTHORITY="$HOME/.Xauthority"; fi
export LG_WD="${LG_WD:-/tmp}"                    # lgpio pipes go to /tmp, not to the project folder

LOG_DIR="$HOME/.cache/poct"
LOG="$LOG_DIR/poct.log"
mkdir -p "$LOG_DIR"
if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 1048576 ]; then mv -f "$LOG" "$LOG.1"; fi

notify() {   # message to the user if a tool is available
    if command -v notify-send >/dev/null 2>&1; then notify-send -i dialog-information "Intelligent POCT" "$1" || true; fi
}

# a single copy of the application only (double-click does not open two windows / fight for the camera)
exec 9>"$LOG_DIR/poct.lock"
if ! flock -n 9; then notify "The application is already running."; exit 0; fi

if [ -f "$APP_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$APP_DIR/.venv/bin/activate"
fi

{ echo; echo "=== $(date '+%Y-%m-%d %H:%M:%S') started ==="; } >> "$LOG"
notify "Loading the model… the window appears in a few seconds."
python DiagnosticsV2.py >> "$LOG" 2>&1
status=$?
echo "=== $(date '+%Y-%m-%d %H:%M:%S') finished (exit code $status) ===" >> "$LOG"
if [ "$status" -ne 0 ]; then notify "The application stopped with an error (exit code $status). See $LOG"; fi
exit "$status"
EOF
chmod +x "$LAUNCHER"

# ------------------------------------------------------------------ 2. application icon (created locally, no extra files)
mkdir -p "$APP_DIR/logo"
if [ ! -f "$ICON" ]; then
    PY="$APP_DIR/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"
    ICON_PATH="$ICON" "$PY" - <<'EOF' || echo "WARNING: the icon could not be created; using the default theme icon."
import os
from PIL import Image, ImageDraw, ImageFont
S, K = 256, 4                                    # final size, supersampling factor (anti-aliasing)
N, GOLD, GREEN, AMBER = (18, 38, 63), (176, 141, 87), (46, 139, 111), (224, 160, 48)
im = Image.new("RGBA", (S * K, S * K), (0, 0, 0, 0))
d = ImageDraw.Draw(im)
d.rounded_rectangle([0, 0, S * K - 1, S * K - 1], radius=48 * K, fill=N)
d.rounded_rectangle([6 * K, 6 * K, S * K - 7 * K, S * K - 7 * K], radius=42 * K, outline=GOLD, width=3 * K)
for cx, col in ((84, GREEN), (172, AMBER)):      # two micropad spots: albumin (left) and creatinine (right)
    r = 38
    d.ellipse([(cx - r - 4) * K, (92 - r - 4) * K, (cx + r + 4) * K, (92 + r + 4) * K], fill=(255, 255, 255))
    d.ellipse([(cx - r) * K, (92 - r) * K, (cx + r) * K, (92 + r) * K], fill=col)
font = None
try:
    import matplotlib
    font = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf"), 52 * K)
except Exception:
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        if os.path.exists(p):
            font = ImageFont.truetype(p, 52 * K)
if font is not None:
    d.text((S * K // 2, 190 * K), "UACR", font=font, fill=(255, 255, 255), anchor="mm")
im.resize((S, S), Image.LANCZOS).save(os.environ["ICON_PATH"])
print("Icon created:", os.environ["ICON_PATH"])
EOF
fi
ICON_FIELD="$ICON"; [ -f "$ICON" ] || ICON_FIELD="applications-science"

# ------------------------------------------------------------------ 3. .desktop files
make_desktop() {
cat > "$1" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=$APP_NAME
GenericName=UACR Classifier
Comment=UACR classification (NORMAL / MICRO / MACRO) from albumin-creatinine micropads
Exec=$LAUNCHER
Path=$APP_DIR
Icon=$ICON_FIELD
Terminal=false
Categories=Science;Education;MedicalSoftware;
StartupNotify=true
EOF
chmod +x "$1"
command -v gio >/dev/null 2>&1 && gio set "$1" metadata::trusted true 2>/dev/null || true
}

mkdir -p "$MENU_DIR" "$DESKTOP_DIR"
make_desktop "$MENU_DIR/poct.desktop"
make_desktop "$DESKTOP_DIR/poct.desktop"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$MENU_DIR" 2>/dev/null || true

if [ "${1:-}" = "--autostart" ]; then
    mkdir -p "$AUTOSTART_DIR"
    make_desktop "$AUTOSTART_DIR/poct.desktop"
    echo "Autostart enabled: the application starts automatically at desktop login."
fi

echo "Done."
echo "  Launcher : $LAUNCHER"
echo "  Menu     : $MENU_DIR/poct.desktop"
echo "  Desktop  : $DESKTOP_DIR/poct.desktop"
echo "  Log      : $HOME/.cache/poct/poct.log"
echo "If the Desktop icon shows a warning sign: right-click -> 'Allow Launching'."

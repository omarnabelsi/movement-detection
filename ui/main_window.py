"""
MotionScope — main application window.

Modern dark sidebar layout with a single QThread worker that captures,
detects motion, tracks objects, and emits annotated frames to the UI.
"""
import os
import traceback
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QStatusBar,
    QFileDialog, QSpinBox, QFrame, QMessageBox, QAction, QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread

from detection.motion_detector import MotionDetector
from detection.tracker import CentroidTracker
from sources.camera import CameraSource
from sources.video import VideoSource
from sources.screen import ScreenSource
from utils.fps_controller import FPSController, FPSCounter
from utils.config import save_config
from ui.video_widget import VideoWidget


# ═══════════════════════════════════════════════════════════════════════
# Dark theme stylesheet
# ═══════════════════════════════════════════════════════════════════════

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QLabel { color: #8b949e; font-size: 11px; letter-spacing: 0.5px; }
QComboBox {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 6px; color: #c9d1d9; padding: 5px 10px;
    font-size: 12px; min-height: 28px;
}
QComboBox::drop-down { border: none; padding-right: 6px; }
QComboBox QAbstractItemView {
    background: #161b22; border: 1px solid #30363d;
    color: #c9d1d9; selection-background-color: #1f6feb;
}
QSpinBox {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 6px; color: #c9d1d9; padding: 5px 8px;
    font-size: 12px; min-height: 28px;
}
QSpinBox::up-button, QSpinBox::down-button {
    background: #21262d; border: none; width: 18px;
}
QStatusBar {
    background: #161b22; border-top: 1px solid #21262d;
    color: #8b949e; font-size: 11px; padding: 2px 8px;
}
QMenuBar {
    background: #161b22; border-bottom: 1px solid #21262d;
    color: #c9d1d9; padding: 2px 4px;
}
QMenuBar::item:selected { background: #21262d; border-radius: 4px; }
QMenu {
    background: #161b22; border: 1px solid #30363d; color: #c9d1d9;
}
QMenu::item:selected { background: #1f6feb; }
"""


# ═══════════════════════════════════════════════════════════════════════
# Worker thread — capture + detect + track + annotate
# ═══════════════════════════════════════════════════════════════════════

class WorkerThread(QThread):
    """Background thread that grabs frames, runs detection and tracking,
    draws annotations, and emits annotated frames to the UI."""

    frame_ready = pyqtSignal(np.ndarray, dict)
    error = pyqtSignal(str)

    def __init__(self, source, detector, tracker, fps_ctrl, resize_width=640,
                 cs2_detector=None, cs2_pipeline=None):
        super().__init__()
        self._source = source
        self._detector = detector
        self._tracker = tracker
        self._fps_ctrl = fps_ctrl
        self._resize_w = resize_width
        self._cs2_det = cs2_detector
        self._cs2_pipeline = cs2_pipeline
        self._running = False
        self._paused = False

    # ── Thread entry ─────────────────────────────────────────────────

    def run(self):
        self._running = True
        counter = FPSCounter()

        while self._running:
            try:
                # Respect pause state
                if self._paused:
                    self.msleep(50)
                    continue

                # FPS limiting via time.sleep
                self._fps_ctrl.tick()

                # Capture one frame
                ok, frame = self._source.get_frame()
                if not ok or frame is None:
                    self.msleep(10)
                    continue

                # Resize for processing performance
                if self._resize_w > 0:
                    h, w = frame.shape[:2]
                    if w != self._resize_w:
                        r = self._resize_w / w
                        frame = cv2.resize(
                            frame, (self._resize_w, int(h * r)),
                            interpolation=cv2.INTER_AREA,
                        )

                # ── Detection ────────────────────────────────────────
                if self._cs2_pipeline is not None:
                    # YOLO-based pipeline (handles detection + heuristics + tracking + annotation)
                    frame, info = self._cs2_pipeline.process_frame(frame)
                    counter.tick()
                    info["fps"] = counter.fps
                    self.frame_ready.emit(frame, info)
                    continue

                cs2 = self._cs2_det is not None
                if cs2:
                    bboxes = self._cs2_det.detect(frame)
                else:
                    _mask, bboxes = self._detector.detect(frame)

                # ── Tracking ─────────────────────────────────────────
                tracked = self._tracker.update(bboxes)

                # ── Annotations ──────────────────────────────────────
                box_clr = (0, 165, 255) if cs2 else (0, 255, 0)   # orange / green
                for oid, (x, y, bw, bh) in tracked.items():
                    # Bounding box
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_clr, 2)
                    # Label with filled background
                    label = f"BOT {oid}" if cs2 else f"ID {oid}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
                    )
                    ly = max(y - th - 8, 0)
                    cv2.rectangle(
                        frame, (x, ly), (x + tw + 6, ly + th + 6), box_clr, -1,
                    )
                    cv2.putText(
                        frame, label, (x + 3, ly + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
                    )

                # Status overlay in top-left corner
                n = len(tracked)
                if cs2:
                    status = f"CS2: {n} bot{'s' if n != 1 else ''}"
                else:
                    status = f"Tracking: {n} object{'s' if n != 1 else ''}"
                (sw, sh), _ = cv2.getTextSize(
                    status, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2,
                )
                cv2.rectangle(frame, (8, 8), (sw + 16, sh + 18), (0, 0, 0), -1)
                color = box_clr if n else (140, 140, 140)
                cv2.putText(
                    frame, status, (12, sh + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
                )

                # FPS measurement
                counter.tick()

                self.frame_ready.emit(frame, {
                    "motion_count": len(bboxes),
                    "tracked_count": n,
                    "fps": counter.fps,
                    "cs2_mode": cs2,
                })
            except Exception:
                self._running = False
                self.error.emit(traceback.format_exc())
                return

    # ── Control ──────────────────────────────────────────────────────

    def stop(self):
        self._running = False
        self.wait(3000)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def set_fps(self, value: int):
        self._fps_ctrl.set_fps(value)

    def set_min_area(self, value: int):
        area = max(0, int(value))
        if self._detector is not None:
            self._detector.min_area = area
        if self._cs2_pipeline is not None:
            self._cs2_pipeline.set_min_box_area(area)


# ═══════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """MotionScope main window — dark sidebar + full-width video feed."""

    def __init__(self, config: dict, config_path: str = "config.yaml"):
        super().__init__()
        self._cfg = config
        self._cfg_path = config_path
        self._worker = None
        self._source = None
        self._detector = None
        self._tracker = None
        self._running = False
        self._last_info: dict = {}

        self._build_ui()
        self._build_menu()

    # ══════════════════════════════════════════════════════════════════
    #  UI construction
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.setWindowTitle("MotionScope")
        self.setMinimumSize(960, 680)
        self.resize(1200, 800)
        self.setStyleSheet(STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left sidebar
        root.addWidget(self._build_sidebar())

        # Video area
        vid_box = QWidget()
        vid_box.setStyleSheet("background: #090c10;")
        vl = QVBoxLayout(vid_box)
        vl.setContentsMargins(0, 0, 0, 0)
        self.video_widget = VideoWidget()
        vl.addWidget(self.video_widget)
        root.addWidget(vid_box, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready · Select a source and press Start")

        # Periodic stats refresh
        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh_stats)
        self._timer.start(400)

        self._on_source_changed(self.source_combo.currentIndex())

    # ── Sidebar ──────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(240)
        w.setStyleSheet(
            "QWidget { background: #0d1117; border-right: 1px solid #21262d; }"
        )
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 20, 16, 20)
        lay.setSpacing(6)

        # Title
        title = QLabel("MOTIONSCOPE")
        title.setStyleSheet(
            "color: #58a6ff; font-size: 15px; font-weight: 700;"
            "letter-spacing: 2px; padding-bottom: 4px;"
        )
        lay.addWidget(title)
        sub = QLabel("Real-time motion detection")
        sub.setStyleSheet("color: #484f58; font-size: 10px; padding-bottom: 12px;")
        lay.addWidget(sub)
        lay.addWidget(self._divider())

        # ── SOURCE ───────────────────────────────────────────────────
        lay.addWidget(self._section("SOURCE"))

        lay.addWidget(self._field("Input"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Camera", "Screen Capture", "Video File"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        lay.addWidget(self.source_combo)

        lay.addSpacing(8)
        lay.addWidget(self._field("Monitor"))
        self.screen_combo = QComboBox()
        self._load_monitors()
        lay.addWidget(self.screen_combo)

        self.native_video_check = QCheckBox("Keep video native resolution")
        self.native_video_check.setStyleSheet(
            "QCheckBox { color: #c9d1d9; font-size: 12px; spacing: 8px; }"
            "QCheckBox::indicator { width: 16px; height: 16px;"
            " background: #161b22; border: 1px solid #30363d; border-radius: 3px; }"
            "QCheckBox::indicator:checked { background: #1f6feb; border-color: #1f6feb; }"
        )
        self.native_video_check.setChecked(
            self._cfg.get("performance", {}).get("keep_video_native_resolution", True)
        )
        self.native_video_check.toggled.connect(self._toggle_native_video)
        lay.addWidget(self.native_video_check)

        lay.addSpacing(6)
        self.btn_video = self._btn("+ Add Video File")
        self.btn_video.clicked.connect(self._choose_video)
        lay.addWidget(self.btn_video)

        # Set default from config
        src_type = self._cfg.get("source", {}).get("type", "screen")
        src_idx = {"camera": 0, "screen": 1, "video": 2}.get(src_type, 1)
        self.source_combo.setCurrentIndex(src_idx)

        lay.addSpacing(12)
        lay.addWidget(self._divider())

        # ── PLAYBACK ─────────────────────────────────────────────────
        lay.addWidget(self._section("PLAYBACK"))

        self.btn_start = self._btn("▶  Start", accent=True)
        self.btn_start.clicked.connect(self._on_start)
        lay.addWidget(self.btn_start)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_pause = self._btn("⏸  Pause", small=True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause)
        row.addWidget(self.btn_pause)
        self.btn_resume = self._btn("⏵  Resume", small=True)
        self.btn_resume.setEnabled(False)
        self.btn_resume.clicked.connect(self._on_resume)
        row.addWidget(self.btn_resume)
        lay.addLayout(row)

        self.btn_stop = self._btn("■  Stop", danger=True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        lay.addWidget(self.btn_stop)

        lay.addSpacing(12)
        lay.addWidget(self._divider())

        # ── SETTINGS ─────────────────────────────────────────────────
        lay.addWidget(self._section("SETTINGS"))

        lay.addWidget(self._field("Target FPS"))
        fps_row = QHBoxLayout()
        fps_row.setSpacing(8)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(
            self._cfg.get("performance", {}).get("target_fps", 30)
        )
        fps_row.addWidget(self.fps_spin)
        b = self._btn("Set", small=True)
        b.setFixedWidth(50)
        b.clicked.connect(self._apply_fps)
        fps_row.addWidget(b)
        lay.addLayout(fps_row)

        lay.addSpacing(4)
        lay.addWidget(self._field("Min Detection Area"))
        area_row = QHBoxLayout()
        area_row.setSpacing(8)
        self.area_spin = QSpinBox()
        self.area_spin.setRange(50, 10000)
        self.area_spin.setSingleStep(50)
        self.area_spin.setValue(
            self._cfg.get("detection", {}).get("min_area", 500)
        )
        area_row.addWidget(self.area_spin)
        b2 = self._btn("Set", small=True)
        b2.setFixedWidth(50)
        b2.clicked.connect(self._apply_area)
        area_row.addWidget(b2)
        lay.addLayout(area_row)

        lay.addSpacing(12)
        lay.addWidget(self._divider())

        # ── GAME MODE ─────────────────────────────────────────────────
        lay.addWidget(self._section("GAME MODE"))
        self.cs2_check = QCheckBox("CS2 Bot Tracking")
        self.cs2_check.setStyleSheet(
            "QCheckBox { color: #c9d1d9; font-size: 12px; spacing: 8px; }"
            "QCheckBox::indicator { width: 16px; height: 16px;"
            " background: #161b22; border: 1px solid #30363d; border-radius: 3px; }"
            "QCheckBox::indicator:checked { background: #1f6feb; border-color: #1f6feb; }"
        )
        lay.addWidget(self.cs2_check)

        self.yolo_check = QCheckBox("Use YOLO Model (AI)")
        self.yolo_check.setStyleSheet(
            "QCheckBox { color: #c9d1d9; font-size: 12px; spacing: 8px; }"
            "QCheckBox::indicator { width: 16px; height: 16px;"
            " background: #161b22; border: 1px solid #30363d; border-radius: 3px; }"
            "QCheckBox::indicator:checked { background: #3fb950; border-color: #3fb950; }"
        )
        lay.addWidget(self.yolo_check)

        lay.addSpacing(4)
        lay.addWidget(self._field("YOLO Model Path"))
        self.model_path_label = QLabel("yolov8n.pt (default)")
        self.model_path_label.setStyleSheet("color: #58a6ff; font-size: 10px;")
        self.model_path_label.setWordWrap(True)
        lay.addWidget(self.model_path_label)
        self.btn_model = self._btn("Select Model", small=True)
        self.btn_model.clicked.connect(self._choose_model)
        lay.addWidget(self.btn_model)

        lay.addSpacing(4)
        lay.addWidget(self._field("Confidence Threshold"))
        conf_row = QHBoxLayout()
        conf_row.setSpacing(8)
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(5, 95)
        self.conf_spin.setSingleStep(5)
        self.conf_spin.setValue(35)
        self.conf_spin.setSuffix("%")
        conf_row.addWidget(self.conf_spin)
        lay.addLayout(conf_row)

        cs2_hint = QLabel("Check YOLO for AI detection · uncheck for motion-based")
        cs2_hint.setStyleSheet("color: #484f58; font-size: 10px;")
        cs2_hint.setWordWrap(True)
        lay.addWidget(cs2_hint)

        lay.addSpacing(12)
        lay.addWidget(self._divider())

        # ── LIVE STATS ───────────────────────────────────────────────
        lay.addWidget(self._section("LIVE STATS"))

        self.lbl_fps = QLabel("FPS  —")
        self.lbl_fps.setStyleSheet(
            "color: #3fb950; font-size: 22px; font-weight: 700; letter-spacing: 1px;"
        )
        lay.addWidget(self.lbl_fps)

        self.lbl_stats = QLabel("No motion")
        self.lbl_stats.setStyleSheet("color: #484f58; font-size: 11px;")
        self.lbl_stats.setWordWrap(True)
        lay.addWidget(self.lbl_stats)

        lay.addStretch()

        ver = QLabel("v2.0 · MOG2 + Centroid Tracker")
        ver.setStyleSheet("color: #21262d; font-size: 10px; padding-top: 8px;")
        lay.addWidget(ver)

        return w

    # ── Widget factories ─────────────────────────────────────────────

    @staticmethod
    def _divider() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setStyleSheet(
            "background: #21262d; min-height: 1px; max-height: 1px; margin: 6px 0;"
        )
        return f

    @staticmethod
    def _section(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #484f58; font-size: 10px; font-weight: 700;"
            "letter-spacing: 1.5px; padding: 4px 0 2px 0;"
        )
        return lbl

    @staticmethod
    def _field(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #8b949e; font-size: 11px; padding: 2px 0 1px 0;")
        return lbl

    @staticmethod
    def _btn(text, accent=False, danger=False, small=False) -> QPushButton:
        btn = QPushButton(text)
        h = "30px" if small else "36px"
        fs = "11px" if small else "12px"
        if accent:
            s = (
                f"QPushButton {{ background:#1f6feb; color:#fff; border:none;"
                f"border-radius:6px; font-size:{fs}; font-weight:600;"
                f"min-height:{h}; padding:0 12px; }}"
                f"QPushButton:hover {{ background:#388bfd; }}"
                f"QPushButton:pressed {{ background:#1158c7; }}"
                f"QPushButton:disabled {{ background:#21262d; color:#484f58; }}"
            )
        elif danger:
            s = (
                f"QPushButton {{ background:#21262d; color:#f85149;"
                f"border:1px solid #f85149; border-radius:6px; font-size:{fs};"
                f"font-weight:600; min-height:{h}; padding:0 12px; }}"
                f"QPushButton:hover {{ background:#3d1c1c; }}"
                f"QPushButton:disabled {{ border-color:#30363d; color:#484f58; }}"
            )
        else:
            s = (
                f"QPushButton {{ background:#21262d; color:#c9d1d9;"
                f"border:1px solid #30363d; border-radius:6px; font-size:{fs};"
                f"font-weight:500; min-height:{h}; padding:0 12px; }}"
                f"QPushButton:hover {{ background:#30363d; border-color:#8b949e; }}"
                f"QPushButton:disabled {{ color:#484f58; border-color:#21262d; }}"
            )
        btn.setStyleSheet(s)
        return btn

    # ── Menu ─────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        a = QAction("&Open Video…", self)
        a.setShortcut("Ctrl+O")
        a.triggered.connect(self._choose_video)
        fm.addAction(a)
        fm.addSeparator()
        a = QAction("&Start", self)
        a.setShortcut("Ctrl+S")
        a.triggered.connect(self._on_start)
        fm.addAction(a)
        a = QAction("S&top", self)
        a.setShortcut("Ctrl+T")
        a.triggered.connect(self._on_stop)
        fm.addAction(a)
        fm.addSeparator()
        a = QAction("E&xit", self)
        a.setShortcut("Ctrl+Q")
        a.triggered.connect(self.close)
        fm.addAction(a)

        hm = mb.addMenu("&Help")
        a = QAction("&About", self)
        a.triggered.connect(self._about)
        hm.addAction(a)

    # ══════════════════════════════════════════════════════════════════
    #  Monitors
    # ══════════════════════════════════════════════════════════════════

    def _load_monitors(self):
        self.screen_combo.clear()
        monitors = ScreenSource.get_available_monitors()
        if not monitors:
            self.screen_combo.addItem("Display 1", 1)
            return
        for m in monitors:
            self.screen_combo.addItem(
                f"Display {m['index']}  ({m['width']}×{m['height']})"
                f"  [{m['left']},{m['top']}]",
                m["index"],
            )

    # ══════════════════════════════════════════════════════════════════
    #  Source controls
    # ══════════════════════════════════════════════════════════════════

    def _on_source_changed(self, idx):
        self.btn_video.setEnabled(idx == 2)
        self.screen_combo.setEnabled(idx == 1 and not self._running)
        self.native_video_check.setEnabled(idx == 2 and not self._running)

    def _choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)",
        )
        if path:
            self._cfg.setdefault("source", {})["video_path"] = path
            save_config(self._cfg, self._cfg_path)
            self.source_combo.setCurrentIndex(2)
            self.status_bar.showMessage(f"Video selected: {path}")

    # ══════════════════════════════════════════════════════════════════
    #  Playback
    # ══════════════════════════════════════════════════════════════════

    def _on_start(self):
        if self._running:
            return

        try:

            # ── Create the video source ──────────────────────────────
            src_idx = self.source_combo.currentIndex()

            if src_idx == 0:  # Camera
                cam_idx = self._cfg.get("source", {}).get("camera_index", 0)
                self._source = CameraSource(index=cam_idx)
                if not self._source.open():
                    QMessageBox.warning(self, "Error", "Cannot open camera.")
                    return

            elif src_idx == 1:  # Screen
                mon_idx = self.screen_combo.currentData()
                if mon_idx is None:
                    mon_idx = 1
                mon_idx = self._pick_capture_monitor(int(mon_idx))
                self._source = ScreenSource(monitor_index=mon_idx)
                if not self._source.open():
                    QMessageBox.warning(self, "Error", "Cannot start screen capture.")
                    return

            elif src_idx == 2:  # Video file
                path = self._cfg.get("source", {}).get("video_path", "")
                if not path:
                    self._choose_video()
                    path = self._cfg.get("source", {}).get("video_path", "")
                if not path:
                    QMessageBox.warning(self, "Error", "No video file selected.")
                    return
                self._source = VideoSource(path)
                if not self._source.open():
                    QMessageBox.warning(self, "Error", f"Cannot open: {path}")
                    return

            # ── Create detector + tracker ────────────────────────────
            det = self._cfg.get("detection", {})
            self._detector = MotionDetector(
                history=det.get("history", 500),
                var_threshold=det.get("var_threshold", 25),
                min_area=det.get("min_area", 500),
                blur_size=det.get("blur_size", 21),
                erode_iterations=det.get("erode_iterations", 1),
                dilate_iterations=det.get("dilate_iterations", 3),
            )
            trk = self._cfg.get("tracker", {})
            self._tracker = CentroidTracker(
                max_disappeared=trk.get("max_disappeared", 30),
            )

            perf = self._cfg.get("performance", {})
            fps_ctrl = FPSController(target_fps=perf.get("target_fps", 30))
            resize_w = perf.get("resize_width", 640)

            # Keep original video resolution by default to preserve quality.
            keep_video_native = perf.get("keep_video_native_resolution", True)
            if src_idx == 2 and keep_video_native:
                resize_w = 0

            # ── Start worker thread ──────────────────────────────────
            cs2_det = None
            cs2_pipeline = None

            if self.cs2_check.isChecked():
                if self.yolo_check.isChecked():
                    # YOLO AI pipeline
                    from cs2_detector_pipeline.pipeline import CS2DetectorPipeline
                    from detector.cs2_heuristics import CS2Heuristics
                    from active_learning.loop import ActiveLearningLoop

                    model_path = getattr(self, "_yolo_model_path", "yolov8n.pt")
                    conf = self.conf_spin.value() / 100.0
                    heuristics = CS2Heuristics(
                        min_box_h=20, max_box_h=600,
                        min_box_area=det.get("min_area", 500),
                        min_aspect=0.15, max_aspect=0.80,
                    )
                    al_loop = ActiveLearningLoop(output_dir="data/active_learning")
                    cs2_pipeline = CS2DetectorPipeline(
                        model_path=model_path,
                        conf=conf,
                        iou=0.45,
                        heuristics=heuristics,
                        active_learning=al_loop,
                    )
                    self.status_bar.showMessage("Loading YOLO model…")
                else:
                    # Legacy motion-based CS2 detection
                    from detection.cs2_bot_detector import CS2BotDetector
                    cs2_det = CS2BotDetector(self._detector, use_color=True)

            self._worker = WorkerThread(
                self._source, self._detector, self._tracker, fps_ctrl, resize_w,
                cs2_detector=cs2_det,
                cs2_pipeline=cs2_pipeline,
            )
            self._worker.frame_ready.connect(self._on_frame)
            self._worker.error.connect(self._on_worker_error)
            self._worker.start()

            self._running = True
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_pause.setEnabled(True)
            self.btn_resume.setEnabled(False)
            self.source_combo.setEnabled(False)
            self.screen_combo.setEnabled(False)

            names = ["Camera", "Screen", "Video"]
            suffix = ""
            if src_idx == 2 and resize_w == 0:
                suffix = " · Native video resolution"
            self.status_bar.showMessage(f"Running · Source: {names[src_idx]}{suffix}")
        except Exception as e:
            if self._source:
                self._source.release()
                self._source = None
            self._running = False
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.btn_resume.setEnabled(False)
            self.source_combo.setEnabled(True)
            self.screen_combo.setEnabled(True)
            QMessageBox.critical(self, "Start Failed", f"{e}\n\n{traceback.format_exc()}")

    def _on_stop(self):
        if not self._running:
            return
        self._running = False

        if self._worker:
            self._worker.stop()
            self._worker = None

        if self._source:
            self._source.release()
            self._source = None

        cv2.destroyAllWindows()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.source_combo.setEnabled(True)
        self.screen_combo.setEnabled(True)
        self.video_widget.clear_display()
        self.status_bar.showMessage("Stopped")

    def _on_pause(self):
        if self._worker:
            self._worker.pause()
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        self.status_bar.showMessage("Paused")

    def _on_resume(self):
        if self._worker:
            self._worker.resume()
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.status_bar.showMessage("Resumed")

    # ── Frame handling ───────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray, info: dict):
        self._last_info = info
        self.video_widget.update_frame(frame)

    def _on_worker_error(self, details: str):
        self._running = False
        if self._source:
            self._source.release()
            self._source = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.source_combo.setEnabled(True)
        self.screen_combo.setEnabled(True)
        self.status_bar.showMessage("Worker crashed")
        QMessageBox.critical(self, "Runtime Error", details)

    def _refresh_stats(self):
        info = self._last_info
        if not self._running or not info:
            self.lbl_fps.setText("FPS  —")
            self.lbl_fps.setStyleSheet(
                "color: #484f58; font-size: 22px; font-weight: 700;"
            )
            return

        fps = info.get("fps", 0)
        self.lbl_fps.setText(f"FPS  {fps:.1f}")
        if fps >= 15:
            c = "#3fb950"
        elif fps >= 8:
            c = "#d29922"
        else:
            c = "#f85149"
        self.lbl_fps.setStyleSheet(
            f"color: {c}; font-size: 22px; font-weight: 700;"
        )

        mc  = info.get("motion_count", info.get("filtered_detections", 0))
        tc  = info.get("tracked_count", 0)
        cs2 = info.get("cs2_mode", False)
        if tc:
            if cs2:
                self.lbl_stats.setText(f"{tc} bot{'s' if tc != 1 else ''} tracked")
                self.lbl_stats.setStyleSheet("color: #ff8c00; font-size: 11px;")
            else:
                self.lbl_stats.setText(
                    f"{mc} motion region{'s' if mc != 1 else ''} · {tc} tracked"
                )
                self.lbl_stats.setStyleSheet("color: #3fb950; font-size: 11px;")
        else:
            self.lbl_stats.setText("No bots" if cs2 else "No motion")
            self.lbl_stats.setStyleSheet("color: #484f58; font-size: 11px;")

        if cs2:
            self.status_bar.showMessage(
                f"CS2 Mode · Bots: {tc} · FPS: {fps:.1f}"
            )
        else:
            self.status_bar.showMessage(
                f"Motion: {mc} · Tracked: {tc} · FPS: {fps:.1f}"
            )

    # ── Settings ─────────────────────────────────────────────────────

    def _apply_fps(self):
        v = self.fps_spin.value()
        self._cfg.setdefault("performance", {})["target_fps"] = v
        save_config(self._cfg, self._cfg_path)
        if self._worker:
            self._worker.set_fps(v)
        self.status_bar.showMessage(f"Target FPS cap → {v}")

    def _apply_area(self):
        v = self.area_spin.value()
        self._cfg.setdefault("detection", {})["min_area"] = v
        save_config(self._cfg, self._cfg_path)
        if self._worker:
            self._worker.set_min_area(v)
        elif self._detector:
            self._detector.min_area = v
        self.status_bar.showMessage(f"Min detection area → {v}")

    def _toggle_native_video(self, checked: bool):
        self._cfg.setdefault("performance", {})["keep_video_native_resolution"] = bool(checked)
        save_config(self._cfg, self._cfg_path)
        mode = "ON" if checked else "OFF"
        self.status_bar.showMessage(f"Native video resolution: {mode}")

    def _choose_model(self):
        """Let the user select a trained YOLO .pt model file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "",
            "PyTorch Weights (*.pt);;All Files (*)",
        )
        if path:
            self._yolo_model_path = path
            name = os.path.basename(path)
            self.model_path_label.setText(name)
            self.status_bar.showMessage(f"Model selected: {name}")

    # ── Monitor selection (Option-1: capture opposite monitor) ───────

    def _pick_capture_monitor(self, requested: int) -> int:
        """If the app window sits on the requested monitor, switch to
        the other one to avoid a mirror-loop."""
        monitors = ScreenSource.get_available_monitors()
        if len(monitors) < 2:
            return requested
        app_mon = self._get_app_monitor()
        if requested != app_mon:
            return requested
        for m in monitors:
            if m["index"] != app_mon:
                return m["index"]
        return requested

    def _get_app_monitor(self) -> int:
        monitors = ScreenSource.get_available_monitors()
        if not monitors:
            return 1
        c = self.frameGeometry().center()
        cx, cy = c.x(), c.y()
        for m in monitors:
            if (m["left"] <= cx < m["left"] + m["width"]
                    and m["top"] <= cy < m["top"] + m["height"]):
                return m["index"]
        return monitors[0]["index"]

    # ── About ────────────────────────────────────────────────────────

    def _about(self):
        QMessageBox.about(
            self,
            "About MotionScope",
            "<h3>MotionScope v3.0 — CS2 Bot Detector</h3>"
            "<p>Professional CS2 bot detection and tracking system.</p>"
            "<ul>"
            "<li>YOLOv8 deep learning object detection</li>"
            "<li>SORT multi-object tracking with Kalman prediction</li>"
            "<li>CS2-specific heuristic filters</li>"
            "<li>Active learning for continuous improvement</li>"
            "<li>Legacy MOG2 motion detection fallback</li>"
            "<li>Camera, screen capture, and video file sources</li>"
            "</ul>"
            "<p>Built with Ultralytics + OpenCV + PyQt5.</p>",
        )

    # ── Keyboard + window close ──────────────────────────────────────

    def keyPressEvent(self, event):
        """ESC key stops capture and closes the window."""
        if event.key() == Qt.Key_Escape:
            self._on_stop()
            self.close()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        """Ensure clean shutdown on window close."""
        self._on_stop()
        event.accept()

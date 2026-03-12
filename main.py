"""
MotionScope — Real-time motion detection and object tracking.

Entry point. Creates the PyQt5 application, loads configuration,
and opens the main window.
"""
import sys
import os
import signal

from utils.config import load_config


def _preload_torch_on_windows() -> None:
    """Load torch before Qt to avoid Windows DLL init-order issues.

    Some Windows setups fail with WinError 1114 when torch is imported
    after QApplication is initialized.
    """
    if os.name != "nt":
        return
    try:
        import torch  # noqa: F401
    except Exception as e:
        # Keep app usable for non-YOLO modes; Start handler now shows details.
        print(f"[startup] torch preload warning: {e}")


def main():
    _preload_torch_on_windows()

    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from ui.main_window import MainWindow

    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("MotionScope")

    # Load configuration
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml",
    )
    config = load_config(config_path)

    # Create and show main window
    window = MainWindow(config, config_path)
    window.show()

    # Allow Ctrl+C in terminal to close the app
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

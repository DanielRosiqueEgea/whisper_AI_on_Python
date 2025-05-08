from PySide6.QtWidgets import QApplication
from gui.windows.main_window import MainWindow

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
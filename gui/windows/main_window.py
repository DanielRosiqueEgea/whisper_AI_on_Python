from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QFileDialog, QListWidgetItem, QLabel,
    QScrollArea
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon, QFontMetrics
import os
from gui.widgets.CustomQbutton import CustomQButton

class FileItemWidget(QWidget):
    def __init__(self, file_path, main_window):
        super().__init__()
        self._estado_resize = None
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.main_window = main_window

        self.label = QLabel(file_path)
        self.edit_button = QPushButton()
        self.delete_button = QPushButton()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.edit_button)
        layout.addWidget(self.delete_button)

        self.edit_button.clicked.connect(self.editar_nombre)
        self.delete_button.clicked.connect(self.borrar_item)
        self.edit_button.setText("Editar")
        self.edit_button.setFixedWidth(80)
        self.delete_button.setText("Borrar")
        self.delete_button.setFixedWidth(80)


        self.actualizar_texto_label()

    def actualizar_texto_label(self,disponible = None):
        # Define un umbral arbitrario para cambiar de texto
        if disponible is None:
            disponible = self.width()
        
        font_metrics = QFontMetrics(self.label.font())

        text_path = self.file_path
        text_name = self.file_name
        text_base = os.path.basename(self.file_path)

        width_path = font_metrics.horizontalAdvance(text_path)
        width_name = font_metrics.horizontalAdvance(text_name)
        width_base = font_metrics.horizontalAdvance(text_base)
        
        # Espacio adicional para los botones
        botones_ancho_total = self.edit_button.width() + self.delete_button.width() + 30
        print(f'Espacio disponible: {disponible} \n tamaño path: {width_path + botones_ancho_total} \n tamaño name: {width_name + botones_ancho_total}')

        
        if disponible <  (width_name + botones_ancho_total):
            nuevo_estado = "icons"    
        elif disponible < (width_path + botones_ancho_total):
            nuevo_estado = "name"
        else:
            nuevo_estado = "completo"

        if nuevo_estado != self._estado_resize:
            if nuevo_estado == 'icons':
                self.label.setText(os.path.basename(self.file_path))
                self.edit_button.setText(u"\u270E")
                self.edit_button.setFixedWidth(30)
                self.delete_button.setText(u"\u2421")
                self.delete_button.setFixedWidth(30)
            elif nuevo_estado == 'name':
                self.label.setText(self.file_name)
                self.edit_button.setText("Editar")
                self.edit_button.setFixedWidth(60)
                self.delete_button.setText("Borrar")
                self.delete_button.setFixedWidth(60)
            else:
                self.label.setText(self.file_path)
                self.edit_button.setText("Editar")
                self.edit_button.setFixedWidth(80)
                self.delete_button.setText("Borrar")
                self.delete_button.setFixedWidth(80)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.actualizar_texto_label()

    def editar_nombre(self):
        nuevo_nombre, ok = QFileDialog.getOpenFileName(self, "Editar nombre", self.file_path)
        if ok and nuevo_nombre:
            self.file_path = nuevo_nombre
            self.label.setText(nuevo_nombre)
            self.main_window.update_archivo(self, nuevo_nombre)

    def borrar_item(self):
        self.main_window.borrar_widget(self)

   
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._estado_resize = None  # Variable para almacenar el estado actual del resize
        self.setWindowTitle("Gestor de Archivos Multimedia")
        self.resize(800, 600)
        self.setMinimumWidth(370)

        self.archivos_seleccionados = []
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.base_dir, ".."))

        # Contenedor principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Menú izquierdo
        self.menu_layout = QVBoxLayout()
        self.btn_open_files = QPushButton("Abrir archivos")

        
       
        self.btn_open_files.clicked.connect(self.abrir_archivos)
        self.menu_layout.addWidget(self.btn_open_files)
        self.menu_layout.addStretch()

        # Scroll de widgets
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_content)
        
        # Agregar layouts al layout principal
        main_layout.addLayout(self.menu_layout, 1)  # Menú ocupa 1 parte
        
        main_layout.addWidget(self.scroll_area, 3)    # Lista ocupa 3 partes

  
    def abrir_archivos(self):
        filtros = "Videos y Audios (*.mp4 *.avi *.mkv *.mp3 *.wav *.flac)"
        archivos, _ = QFileDialog.getOpenFileNames(self, "Seleccionar archivos", "", filtros)
        
        for archivo in archivos:
            if archivo not in self.archivos_seleccionados:
                self.archivos_seleccionados.append(archivo)
                widget = FileItemWidget(archivo, self)
                self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, widget)

    def update_archivo(self, widget, nuevo_nombre):
        idx = self.scroll_layout.indexOf(widget)
        if idx != -1:
            self.archivos_seleccionados[idx] = nuevo_nombre

    def borrar_widget(self, widget):
        idx = self.scroll_layout.indexOf(widget)
        if idx != -1:
            self.archivos_seleccionados.pop(idx)
            widget.setParent(None)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        disponible = self.width()
        for widget in self.menu_layout.children():
            widget.actualizar_texto(disponible)
        




if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
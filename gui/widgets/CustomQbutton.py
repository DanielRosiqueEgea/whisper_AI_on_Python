from PySide6.QtWidgets import QPushButton
from PySide6.QtGui import QIcon, QFontMetrics
import os     


class CustomQButton(QPushButton):
    def __init__(self, key, textos_dict):
        super().__init__(textos_dict[key]["completo"])
        self.key = key
        self.textos = textos_dict[key]
        self._estado_resize = None

        self.texto_completo = self.textos["completo"]
        self.texto_reducido = self.textos["abreviado"]
        self.icon_path = self.textos["icono"]

    def actualizar_texto(self, disponible=None):
        if disponible is None:
            disponible = self.width()

        font_metrics = QFontMetrics(self.font())
        width_completo = font_metrics.horizontalAdvance(self.texto_completo) + 10
        width_abreviado = font_metrics.horizontalAdvance(self.texto_reducido) + 10
        icon_space = 40

        if disponible < icon_space:
            nuevo_estado = "icono"
        elif disponible < width_abreviado:
            nuevo_estado = "abreviado"
        else:
            nuevo_estado = "completo"

        if nuevo_estado != self._estado_resize:
            self._estado_resize = nuevo_estado

            if nuevo_estado == "icono" and os.path.exists(self.icon_path):
                self.setIcon(QIcon(self.icon_path))
                self.setText("")
                self.setIconSize(QSize(24, 24))
                self.setFixedSize(40, 40)
            elif nuevo_estado == "abreviado":
                self.setIcon(QIcon())
                self.setText(self.texto_reducido)
                self.setFixedWidth(60)
            else:
                self.setIcon(QIcon())
                self.setText(self.texto_completo)
                self.setFixedWidth(120)


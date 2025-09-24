import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import (QPixmap, QImage, QPainter, QColor)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QGraphicsScene, QGraphicsView, QFileDialog, QColorDialog, QGraphicsPixmapItem, QGraphicsEllipseItem, QPushButton, QMessageBox)

import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import matplotlib.pyplot as plt

class Documentation():
    def __init__(self):
        self.list_r = [2, 4, 8, 12]
        self.list_eps = [0.1**2, 0.2**2, 0.3**2, 0.4**2]

    def createGuidedFilterComparisonImage(self, segment):
        rows = len(self.list_r)
        columns = len(self.list_eps)
        # Crea una figura y una cuadrícula de subgráficos (axes)
        # figsize controla el tamaño final de la imagen en pulgadas
        fig, axes = plt.subplots(rows, columns, figsize=(10, 7.5))

        for i, r in enumerate(self.list_r):
            for j, eps in enumerate(self.list_eps):
                segment.setFeatheredMaskedImage(r, eps)
                img = segment.getFeatheredMaskedImageCV2()

                # Muestra la imagen en el subgráfico correspondiente
                ax = axes[i, j]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])

                if i == rows - 1:
                    ax.set_xlabel(f'ε = {eps:.2f}', fontsize=12, rotation=00, labelpad=7, ha='center', va='center')
                
                if j == 0:
                    ax.set_ylabel(f'r = {r}', fontsize=12, rotation=90, labelpad=7, ha='center', va='center')

        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.savefig("images/comparison_image.png", dpi=300, bbox_inches='tight')

    def setListR(self, radius):
        self.list_r = radius
    def setListEPS(self, eps):
        self.list_eps = eps
    def getListR(self):
        return self.list_r
    def getListEPS(self):
        return self.list_eps

class ImageSegment():
    def __init__(self, source_image_path):
        # Attributes
        self.raw_mask = None
        self.score = None
        self.feathered_mask = None
        self.input_point = None
        self.input_label = None
        self.mask_color = np.array([30, 144, 255], dtype=np.uint8)
        self.raw_masked_image = None
        self.feathered_masked_image = None

        # Create SAM2 predictor
        self.device = torch.device("cpu")
        self.sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

        # Load image
        self.image = cv2.cvtColor(cv2.imread(source_image_path), cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.image)

    # Methods
    def getMask(self):
        return self.raw_mask
    
    def getScore(self):
        return self.score
    
    def getMaskColor(self):
        return self.mask_color.tolist()

    def setMaskColor(self, color_RGB):
        self.mask_color = np.array(color_RGB)

    def setInputPointArray(self, input_point_list):
        self.input_point = np.array(input_point_list)
    
    def setInputLabelArray(self, input_label_list):
        self.input_label = np.array(input_label_list)
    
    def setRawMaskedImage(self):
        self.raw_mask, self.score, _ = self.predictor.predict(
                                        point_coords=self.input_point,
                                        point_labels=self.input_label,
                                        multimask_output=False,
                                        )
        self.raw_mask = self.raw_mask[0]
        self.score = self.score[0]
        self.raw_masked_image = self.createColoredMask(self.raw_mask, self.mask_color)
    
    def setFeatheredMaskedImage(self, r, eps):
        self.guidedFilter(r, eps)
        self.feathered_masked_image = self.createColoredMask(self.feathered_mask, self.mask_color+128)

    def getRawMaskedImageCV2(self):
        return self.raw_masked_image

    def getRawMaskedImagePixmap(self):
        return self.fromCV2ToQPixmap(self.raw_masked_image)

    def getFeatheredMaskedImageCV2(self):
        return self.feathered_masked_image

    def getFeatheredMaskedImagePixmap(self):
        return self.fromCV2ToQPixmap(self.feathered_masked_image)

    def guidedFilter(self, radius=15, eps=0.01):
        guide_I = self.image.astype(np.float32)/255.0
        mask_p = self.raw_mask.astype(np.float32)

        self.feathered_mask = cv2.ximgproc.guidedFilter(
                            guide=guide_I,
                            src=mask_p,
                            radius=radius,
                            eps=eps
                        )

    @staticmethod
    def createColoredMask(mask, mask_color):
        color = np.hstack((mask_color/255, [0.4]))
        h, w = mask.shape[-2:]
        masked_image_float = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        masked_image_uint8 = (masked_image_float * 255).astype(np.uint8)
        return masked_image_uint8

    @staticmethod
    def fromCV2ToQPixmap(imgCV2):
        height, width, channel = imgCV2.shape
        bytes_per_line = channel * width
        if channel == 3:
            img_format = QImage.Format.Format_RGB888
        elif channel == 4:
            img_format = QImage.Format.Format_RGBA8888

        qimage = QImage(imgCV2.data, width, height, bytes_per_line, img_format)
        imgQPixmap = QPixmap.fromImage(qimage.copy())

        return imgQPixmap
    
    """
    @staticmethod
    def fromQPixmapToCV2(imgQPixmap: QPixmap):
        qimage = imgQPixmap.toImage()
        if qimage.format() != QImage.Format.Format_ARGB32:
            qimage = qimage.convertToFormat(QImage.Format.Format_ARGB32)
        
        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())
        imgCV2 = np.array(ptr).reshape(qimage.height(), qimage.width(), 4)
        imgCV2 = imgCV2[:, :, :3]

        return imgCV2"""

class Viewer(QGraphicsView):
    # Initialization
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None
        self.mask_item = None
        self.click_pos = None
        self.is_panning = False
        self.scene_rect = None
        self.min_markpoint_radius = 1
        self.point_coordinates = []
        self.point_labels = []
        self.marker_items = []

        self.setRenderHint(QPainter.Antialiasing) # Smooths the edges of drawn points
        self.setRenderHint(QPainter.SmoothPixmapTransform) # Smooths the image when scaling it
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse) # Zoom pointing to the mouse
        self.setDragMode(QGraphicsView.ScrollHandDrag) # Allows scroll and pan
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # Hides horizontal scroll bar
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # Hides vertical scroll bar
        self.viewport().setCursor(Qt.ArrowCursor) # Changes cursor shape

    # Methods
    def setImageFromPath(self, source_img_path):
        pixmap = QPixmap(source_img_path)
        self.setImageFromPixmap(pixmap)

    def setImageFromPixmap(self, pixmap):
        self.scene.clear()
        self.mask_item = None               # <-- Olvida la referencia a la máscara anterior.
        self.point_coordinates.clear()      # <-- Limpia la lista de coordenadas.
        self.point_labels.clear()           # <-- Limpia la lista de etiquetas.
        
        self.pixmap_item = QGraphicsPixmapItem(pixmap)        
        self.scene.addItem(self.pixmap_item)
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.scene_rect = self.scene.itemsBoundingRect()
    
    def addOverlay(self, pixmap: QPixmap):
        """
        Añade un QPixmap como una capa superpuesta sobre la imagen principal.
        Si ya existe una capa anterior, la elimina primero.
        """
        # Si ya había una máscara, la eliminamos de la escena
        if self.mask_item is not None:
            self.scene.removeItem(self.mask_item)

        # Creamos el nuevo item para la máscara
        self.mask_item = QGraphicsPixmapItem(pixmap)
        
        # Le damos un Z-value mayor que 0 para que se dibuje sobre la imagen base
        self.mask_item.setZValue(1)
        
        # Lo añadimos a la escena
        self.scene.addItem(self.mask_item)
    
    def getPointCoordinates(self):
        return self.point_coordinates

    def getPointLabels(self):
        return self.point_labels

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
    
    def wheelEvent(self, event):
        factor_zoom = 1.1
        if event.angleDelta().y() > 0:
            # Zoom in
            factor = factor_zoom
        else:
            # Zoom out
            visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()
            if visible_rect.width() > self.scene_rect.width() or visible_rect.height() > self.scene_rect.height():
                self.fitInView(self.scene_rect, Qt.KeepAspectRatio)
                return
            factor = 1 / factor_zoom

        self.scale(factor, factor)
    
    def enterEvent(self, event):
        self.viewport().setCursor(Qt.ArrowCursor)
        super().enterEvent(event)
    
    def mouseMoveEvent(self, event):
        # Si el botón izquierdo está presionado y tenemos una posición de inicio...
        if event.buttons() == Qt.LeftButton and self.click_pos:
            # Calculamos la distancia desde el punto de inicio
            # manhattanLength es una forma rápida de medir la distancia
            dist = (event.position() - self.click_pos).manhattanLength()
            # QApplication.startDragDistance() es la distancia recomendada por el SO
            # para considerar algo como un arrastre
            if dist > QApplication.startDragDistance():
                self.is_panning = True
        
        # Pasamos el evento a la clase base para que el paneo funcione
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and not self.is_panning:
            if self.pixmap_item:
                coordinates = self.mapToScene(event.position().toPoint()).toPoint()
                self.point_coordinates.append((coordinates.x(), coordinates.y()))
                self.point_labels.append(1)
                # Verificamos si el clic fue DENTRO de la imagen
                if self.pixmap_item.contains(coordinates):
                    # Dibujamos un círculo rojo para marcar el punto
                    radius = (int(self.scene_rect.width())>>8) + self.min_markpoint_radius # Escalar punto a dibujar
                    marker = QGraphicsEllipseItem(
                        coordinates.x() - radius,
                        coordinates.y() - radius,
                        radius * 2,
                        radius * 2,
                    )
                    marker.setBrush(Qt.green)
                    marker.setPen(Qt.NoPen)
                    # Aseguramos que el marcador se dibuje encima de la foto
                    marker.setZValue(1) 
                    self.scene.addItem(marker)
                    self.marker_items.append(marker)
        # Reseteamos el estado para el próximo clic
        self.press_pos = None
        self.is_panning = False
        
        # Pasamos el evento a la clase base para que se complete la lógica del drag
        super().mouseReleaseEvent(event)
        self.viewport().setCursor(Qt.ArrowCursor)
    
    def mousePressEvent(self, event):
        # Solo nos interesa el clic izquierdo para iniciar la lógica
        if event.button() == Qt.LeftButton and self.pixmap_item:
            self.click_pos = event.position()
            self.is_panning = False
        elif event.button() == Qt.RightButton and self.pixmap_item:
            coordinates = self.mapToScene(event.position().toPoint()).toPoint()
            self.point_coordinates.append((coordinates.x(), coordinates.y()))
            self.point_labels.append(0)
            # Verificamos si el clic fue DENTRO de la imagen
            if self.pixmap_item.contains(coordinates):
                # Dibujamos un círculo rojo para marcar el punto
                radius = (int(self.scene_rect.width())>>8) + self.min_markpoint_radius # Escalar punto a dibujar
                marker = QGraphicsEllipseItem(
                    coordinates.x() - radius,
                    coordinates.y() - radius,
                    radius * 2,
                    radius * 2,
                )
                marker.setBrush(Qt.red)
                marker.setPen(Qt.NoPen)
                # Aseguramos que el marcador se dibuje encima de la foto
                marker.setZValue(1) 
                self.scene.addItem(marker)
                self.marker_items.append(marker)
        
        # Pasamos el evento a la clase base para que el drag se inicie si es necesario
        super().mousePressEvent(event)
    
    def clearAllPoints(self):
        if self.marker_items:
            for marker in self.marker_items:
                self.scene.removeItem(marker)

            self.marker_items.clear()
            self.point_coordinates.clear()
            self.point_labels.clear()

            #if self.mask_item:
            #    self.scene.removeItem(self.mask_item)
            #    self.mask_item = None
    
    def clearLastPoint(self):
        if self.marker_items:
            last_point = self.marker_items.pop()
            self.scene.removeItem(last_point)
            self.point_coordinates.pop()
            self.point_labels.pop()

class MainWindow(QMainWindow):
    # Initialization
    def __init__(self):
        super().__init__()

        # Attributes
        self.source_img_path = None
        self.segment = None

        # Window Settings
        self.setWindowTitle("GUI")
        self.resize(1200, 700)

        # Widgets
        self.viewer = Viewer()
        #self.viewer.setImageFromPath("/home/ddgiraldo/Thesis/Test SAM2/images/bac.jpg")

        # Menu Bar
        menu_bar = self.menuBar()

        menu_file = menu_bar.addMenu("Archivo")
        action_open_img = menu_file.addAction("Abrir Imagen")
        action_open_img.triggered.connect(self.openImage)
        submenu_extract = menu_file.addMenu("Extraer")
        action_extract_raw_mask = submenu_extract.addAction("Extraer mascara cruda")
        action_extract_guided_filter = submenu_extract.addAction("Extraer filtro guiado")


        menu_edit = menu_bar.addMenu("Editar")
        action_mask_color = menu_edit.addAction("Cambiar color máscara")
        action_mask_color.triggered.connect(self.openColorDialog)
        action_delete_last_point = menu_edit.addAction("Eliminar el último punto")
        action_delete_last_point.triggered.connect(self.viewer.clearLastPoint)
        action_delete_all_points = menu_edit.addAction("Eliminar todos los puntos")
        action_delete_all_points.triggered.connect(self.viewer.clearAllPoints)
        
        menu_run = menu_bar.addMenu("Correr")
        action_run = menu_run.addAction("Segmentar")
        action_run.triggered.connect(self.runSegmentation)

        # Layouts
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.viewer)

        # Containers and Layouts
        container = QWidget()
        container.setObjectName("MainContainer")
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Styles
        styles = """
        QMenuBar {
            font-size: 14px; /* Tamaño de la fuente para la barra principal (Archivo, Editar, etc.) */
        }
        QMenu {
            font-size: 13px; /* Tamaño de la fuente para los menús desplegables */
        }
        QMenu::item {
            padding: 5px 8px; 
        }
        QMenu::item:selected {
            background-color: #308cc6; /* Un color azul para resaltar */
            color: white;             /* Cambia el color del texto a blanco */
            border: 1px solid #26709e;
        }
        """
        # Aplica la hoja de estilos a toda la ventana
        self.setStyleSheet(styles)

    # Methods
    def openImage(self):
        self.source_img_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "", "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp)"
        )
        if self.source_img_path:
            self.viewer.setImageFromPath(self.source_img_path)
            self.segment = ImageSegment(self.source_img_path)

    def runSegmentation(self):
        if self.source_img_path:
            if self.viewer.point_coordinates:
                self.segment.setInputPointArray(self.viewer.point_coordinates)
                self.segment.setInputLabelArray(self.viewer.point_labels)
                self.segment.setRawMaskedImage()
                self.viewer.addOverlay(self.segment.getRawMaskedImagePixmap())
                #self.segment.setFeatheredMaskedImage(4, 0.01)
                #self.viewer.addOverlay(self.segment.getFeatheredMaskedImagePixmap())
                self.doc = Documentation()
                self.doc.createGuidedFilterComparisonImage(self.segment)
            else:
                QMessageBox.warning(self, 
                                "Prompts no encontrados", 
                                "Por favor, crea los puntos antes de correr la segmentación.")
                if self.viewer.mask_item:
                    self.viewer.scene.removeItem(self.viewer.mask_item)
                    self.viewer.mask_item = None
        else:
            QMessageBox.warning(self, 
                                "Imagen no encontrada", 
                                "Por favor, carga una imagen antes de correr la segmentación.")

    def openColorDialog(self):
        if self.segment:
            r,g,b = self.segment.getMaskColor()
            default_color = QColor.fromRgb(r,g,b)
            new_color = QColorDialog.getColor(default_color, self, "Elige un color")
            if new_color.isValid():
                r = new_color.red()
                g = new_color.green()
                b = new_color.blue()
                self.segment.setMaskColor([r,g,b])
        else:
            QMessageBox.warning(self, 
                                "Imagen no encontrada", 
                                "Por favor, carga una imagen antes de seleccionar el color.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec() #Start the event loop
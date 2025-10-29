import sys
import os
from PySide6.QtCore import Qt
from PySide6.QtGui import (QPixmap, QImage, QPainter, QColor, QPalette)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QGraphicsScene, 
                                QGraphicsView, QFileDialog, QColorDialog, QGraphicsPixmapItem, QGraphicsEllipseItem, 
                                QPushButton, QMessageBox, QFrame, QCheckBox, QGroupBox, QLabel, QLineEdit)

import numpy as np
import pandas as pd
import pickle
import torch
import cv2
from skimage.color import (deltaE_ciede2000, lab2rgb)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

class Documentation():
    def __init__(self) -> None:
        self.list_r = [2, 4, 8, 12]
        self.list_eps = [0.1**2, 0.2**2, 0.3**2, 0.4**2]

    def createGuidedFilterComparisonImage(self, processing):
        rows: int = len(self.list_r)
        columns: int = len(self.list_eps)
        # Crea una figura y una cuadrícula de subgráficos (axes)
        # figsize controla el tamaño final de la imagen en pulgadas
        fig, axes = plt.subplots(rows, columns, figsize=(10, 7.5))

        for i, r in enumerate(self.list_r):
            for j, eps in enumerate(self.list_eps):
                processing.guidedFilter(r, eps)
                img = processing.createColoredMask(processing.getFeatheredMask(),
                                                            processing.getFeatheredMaskColor())

                # Muestra la imagen en el subgráfico correspondiente
                ax = axes[i, j]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])

                if i == rows - 1:
                    ax.set_xlabel(f'ε = {eps:.2f}', fontsize=12, rotation=00, labelpad=7, ha='center', va='center')
                
                if j == 0:
                    ax.set_ylabel(f'r = {r}', fontsize=12, rotation=90, labelpad=7, ha='center', va='center')
        img_path = "resources/exported_images/filter_comparison_image.png"
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        return img_path

    @staticmethod
    def createMaskComparisionImage(img_raw_mask, raw_color, img_feathered_mask, feathered_color):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(img_raw_mask)
        ax.imshow(img_feathered_mask)
        ax.set_title("Máscaras Superpuestas")
        ax.axis('off')
        raw_color = np.hstack((raw_color/255, [0.4]))
        feathered_color = np.hstack((feathered_color/255, [0.4]))
        raw_patch = mpatches.Patch(color=raw_color, label='Máscara cruda')
        feathered_patch = mpatches.Patch(color=feathered_color, label='Máscara filtrada')
        ax.legend(handles=[raw_patch, feathered_patch], loc='upper right')
        img_path = "resources/exported_images/mask_comparison_image.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        return img_path
    
    @staticmethod
    def createHistogramImage(histograms):
        """
        Crea y guarda una imagen del histograma de color a partir de los datos calculados.
        """
        if not histograms:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Histograma de Color del Segmento')
        ax.set_xlabel('Intensidad del Píxel')
        ax.set_ylabel('Cantidad de Píxeles')

        # Dibuja el histograma para cada canal
        for color, hist in histograms.items():
            ax.plot(hist, color=color, label=f'Canal {color.upper()}')
        
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim([0, 256]) # El rango de intensidad de color es de 0 a 255

        img_path = "resources/exported_images/histogram_image.png"
        plt.tight_layout()
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        
        return img_path

    def setListR(self, radius) -> None:
        self.list_r = radius
    def setListEPS(self, eps) -> None:
        self.list_eps = eps
    def getListR(self) -> list[int]:
        return self.list_r
    def getListEPS(self) -> list[float]:
        return self.list_eps

class Calibration():
    def __init__(self) -> None:
        self.color_checker_img_path = "resources/calibration/color_checker.jpg"
        self.params_path = "resources/calibration/calibration_params.pickle"

    def getParamsPath(self):
        return self.params_path
    
    def setColorCheckerPath(self, path) -> None:
        self.color_checker_img_path = path
            
    def detectColorChecker(self):
        image = cv2.imread(self.color_checker_img_path)
        # Create a ColorChecker detector
        detector = cv2.mcc.CCheckerDetector_create()
    
        # Process the image to detect the ColorChecker
        detected = detector.process(image, cv2.mcc.MCC24, 1)
        
        if not detected:
            return None

        # Get the list of detected ColorCheckers
        checkers = detector.getListColorChecker()
        
        for checker in checkers:

            # Get the detected color patches and rearrange them
            chartsRGB = checker.getChartsRGB()
            width, height = chartsRGB.shape[:2]
            src = chartsRGB[:, 1].copy().reshape(int(width / 3), 1, 3) / 255.0
            
            return src
        
        return None
    
    def saveCalibrationParams(self, color_patches) -> None:
        # Save the color patches and configuration to a pickle file
        params = {
            'color_patches': color_patches
        }
        with open(self.params_path, 'wb') as f:
            pickle.dump(params, f)
    
    def reconstructModelFromParams(self):
        # Load the color patches and configuration from a pickle file
        with open(self.params_path, 'rb') as f:
            params = pickle.load(f)

        # Reconstruct the color correction model from parameters
        color_patches = params['color_patches']
        model = cv2.ccm_ColorCorrectionModel(color_patches, cv2.ccm.COLORCHECKER_Macbeth)
        
        # Configure the model
        model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
        model.setCCM_TYPE(cv2.ccm.CCM_3x3)
        model.setDistance(cv2.ccm.DISTANCE_CIE2000)
        model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
        model.setLinearGamma(2.2)
        model.setLinearDegree(3)
        model.setSaturatedThreshold(0, 0.98)
        
        # Run the model
        model.run()
        return model
    
    @staticmethod
    def applyColorCorrection(image, model):
        # Apply color correction to the image
        image = image.astype(np.float64) / 255.0

        # Perform inference with the model
        calibrated_image = model.infer(image)
        out_ = calibrated_image * 255
        out_[out_ < 0] = 0
        out_[out_ > 255] = 255
        out_img = out_.astype(np.uint8)

        # Convert back to BGR
        #out_img = cv2.cvtColor(out_, cv2.COLOR_RGB2BGR)
        return out_img


class ImageProcessing():
    def __init__(self, source_image_path, calibration: Calibration) -> None:
        # Attributes
        self.raw_mask = None
        self.score = None
        self.feathered_mask = None
        self.input_point = None
        self.input_label = None
        self.feathered_mask_color = np.array([158, 16, 127], dtype=np.uint8)
        self.raw_mask_color = 255 - self.feathered_mask_color
        self.r_filter = 2
        self.eps_filter = 0.3**2
        self.main_color = None

        # Load PANTONE database
        self.pantone_database_path="resources/pantone_databases/PANTONE Solid Coated-V4.json"
        self.loadPantoneDatabase()

        # Load image
        self.original_image = cv2.cvtColor(self.cropSquare(cv2.imread(source_image_path)), cv2.COLOR_BGR2RGB)
        model = calibration.reconstructModelFromParams()
        self.original_image = calibration.applyColorCorrection(self.original_image, model)
        self.original_image_size = self.original_image.shape[:2]
        self.scaled_image_size = (720, 720)
        self.scaled_image = self.decimateImage(self.original_image)

        # Create SAM2 predictor
        device = torch.device("cpu")
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.predictor.set_image(self.scaled_image)

    # Methods
    def saveOriginalImage(self, path) -> np.ndarray:
        cv2.imwrite(path, cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR))
    
    def getScaledImage(self) -> np.ndarray:
        return self.scaled_image
        
    def getRawMask(self):
        return self.raw_mask
    
    def getFeatheredMask(self):
        return self.feathered_mask
    
    def getScaledFeatheredMask(self):
        return self.scaled_feathered_mask
    
        # NO USADA
    def getScore(self):
        return self.score

    def getFilterR(self) -> int:
        return self.r_filter
    
    def getFilterEPS(self) -> float:
        return self.eps_filter
    
    def getFeatheredMaskColor(self):
        return self.feathered_mask_color

    def setFeatheredMaskColor(self, color_RGB) -> None:
        self.feathered_mask_color = np.array(color_RGB)
    
    def getRawMaskColor(self):
        return self.raw_mask_color

    def decimateImage(self, image: np.ndarray, is_mask: bool = False) -> np.ndarray:
        if not type:
            method = cv2.INTER_AREA
        else:
            method = cv2.INTER_NEAREST

        return cv2.resize(image, self.scaled_image_size, interpolation=method)
    
    def interpolateImage(self, image: np.ndarray, is_mask: bool = False) -> np.ndarray:
        if not type:
            method = cv2.INTER_CUBIC
        else:
            method = cv2.INTER_NEAREST

        return cv2.resize(image, self.original_image_size, interpolation=method)

    def setInputPointArray(self, input_point_list) -> None:
        self.input_point = np.array(input_point_list)
    
    def setInputLabelArray(self, input_label_list) -> None:
        self.input_label = np.array(input_label_list)
    
    def setRawMask(self) -> None:
        self.raw_mask, self.score, _ = self.predictor.predict(
                                        point_coords=self.input_point,
                                        point_labels=self.input_label,
                                        multimask_output=False,
                                        )
        self.raw_mask = self.interpolateImage(self.raw_mask[0], is_mask=True)
        self.score = self.score[0]

    def guidedFilter(self, radius=15, eps=0.01) -> None:
        guide_I = self.original_image.astype(np.float32)/255.0
        mask_p = self.raw_mask.astype(np.float32)

        self.feathered_mask = cv2.ximgproc.guidedFilter(
                            guide=guide_I,
                            src=mask_p,
                            radius=radius,
                            eps=eps
                        )
        self.scaled_feathered_mask = self.decimateImage(self.feathered_mask)

    def getSegmentedRegionHistogram(self):
        if self.feathered_mask is None or self.original_image is None:
            return None
        
        _, binary_mask_uint8 = cv2.threshold(self.feathered_mask, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask_uint8 = binary_mask_uint8.astype(np.uint8)

        colors = ('b', 'g', 'r') # OpenCV usa el orden BGR
        histograms = {}

        for i, color in enumerate(colors):
            # cv2.calcHist(images, channels, mask, histSize, ranges)
            # - [self.image]: La imagen fuente.
            # - [i]: El canal a calcular (0 para Azul, 1 para Verde, 2 para Rojo).
            # - binary_mask_uint8: La máscara. Solo los píxeles donde la máscara es no-cero se incluyen.
            # - [256]: El número de "bins" o niveles de intensidad (0 a 255).
            # - [0, 256]: El rango de intensidad.
            hist = cv2.calcHist([self.original_image], [i], binary_mask_uint8, [256], [0, 256])
            histograms[color] = hist
            
        return histograms

    def loadPantoneDatabase(self) -> None:
        with open(self.pantone_database_path, "r", encoding='utf-8') as f:
            pantone_db = json.load(f)
            self.pantone_lab_colors = np.array([p["components"] for p in pantone_db["records"].values()], dtype=np.float32)
            self.pantone_name_colors = np.array([p["name"] for p in pantone_db["records"].values()], dtype=np.str_)
    
    def findNearestPantone(self, lab_color: np.array) -> dict:
        delta_Es = deltaE_ciede2000(np.array([lab_color], dtype=np.float32), self.pantone_lab_colors)
        idx = np.argmin(delta_Es)
        name = self.pantone_name_colors[idx]
        lab = self.pantone_lab_colors[idx]
        rgb = np.astype(np.round(lab2rgb(lab)*255), np.uint8)
        return {"Pantone Name": name, "Pantone LAB": lab, "Pantone RGB": rgb, "Delta E": delta_Es[idx]}
    
    def getMainColor(self) -> dict:
        return self.main_color

    def getColorByWeightedMedian(self) -> np.ndarray:
        if self.original_image.size <= 0:
            return None
        # Convertir a LAB en rangos estándar
        image_lab = self.fromRGBtoLAB(self.original_image)

        # Filtrar pixels con peso significativo (> 0 para incluir toda la gradación)
        valid_mask: bool = self.feathered_mask > 0
        pixels_lab = image_lab[valid_mask]
        weights = self.feathered_mask[valid_mask]
        
        # Calcular mediana ponderada para cada canal (L*, a*, b*)
        dominant_color_lab = np.empty(3, dtype=np.float32)
        
        for channel in range(3):
            # Extraer valores del canal actual
            channel_values = pixels_lab[:, channel]
            
            # Ordenar valores y sus pesos correspondientes
            sort_idx = np.argsort(channel_values)
            sorted_values = channel_values[sort_idx]
            sorted_weights = weights[sort_idx]
            
            # Calcular suma acumulativa normalizada de pesos
            cumsum = np.cumsum(sorted_weights)
            cumsum /= cumsum[-1]  # Normalizar a [0, 1]
            
            # Encontrar índice donde suma acumulativa >= 0.5 (mediana ponderada)
            median_idx = np.searchsorted(cumsum, 0.5)
            dominant_color_lab[channel] = sorted_values[median_idx]
        
        self.main_color = self.findNearestPantone(dominant_color_lab)
        self.main_color["Original LAB"] = dominant_color_lab

        return self.main_color

    def getColorByKMeans(n_clusters: int = 3,
                            min_weight_threshold: float = 0.3) -> np.ndarray:
        # Convertir a LAB en rangos estándar
        image_lab = self.fromRGBtoLAB(self.original_image)
        
        # Filtrar pixels con peso significativo
        valid_mask: bool = self.feathered_mask > min_weight_threshold
        pixels_lab = image_lab[valid_mask]
        weights = self.feathered_mask[valid_mask]
        
        # Verificar que hay suficientes pixels
        if len(pixels_lab) < n_clusters * 10:
            # Fallback a media ponderada si hay muy pocos pixels
            weights_norm = weights / np.sum(weights)
            return np.average(pixels_lab, axis=0, weights=weights_norm)

        # Criterios de parada para K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # Ejecutar K-means
        _, labels, centroids = cv2.kmeans(
            pixels_lab.astype(np.float32), 
            n_clusters, 
            None, 
            criteria, 
            10,  # intentos
            cv2.KMEANS_PP_CENTERS
        )
        
        labels = labels.flatten()
        
        # Calcular peso acumulado por cluster
        cluster_weights = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_mask = (labels == i)
            cluster_weights[i] = np.sum(weights[cluster_mask])
        
        # Seleccionar cluster con mayor peso acumulado
        dominant_cluster_idx = np.argmax(cluster_weights)
        dominant_color_lab = centroids[dominant_cluster_idx]
        
        return dominant_color_lab
    
    @staticmethod
    def fromRGBtoLAB(image_rgb: np.ndarray) -> np.ndarray:
        # Paso 1: Convertir RGB a LAB (OpenCV devuelve uint8 en rangos comprimidos)
        image_lab_opencv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        
        # Paso 2: Convertir a float32 para precisión en cálculos
        image_lab = image_lab_opencv.astype(np.float32)
        
        # Paso 3: Ajustar a rangos estándar CIELAB
        image_lab[:, :, 0] = image_lab[:, :, 0] * (100.0 / 255.0)  # L*: [0, 100]
        image_lab[:, :, 1] = image_lab[:, :, 1] - 128.0  # a*: [-128, 127]
        image_lab[:, :, 2] = image_lab[:, :, 2] - 128.0  # b*: [-128, 127]
        
        return image_lab

    @staticmethod
    def createColoredMask(mask, mask_color):
        color = np.hstack((mask_color/255, [0.4]))
        h, w = mask.shape[-2:]
        masked_image_float = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        masked_image_uint8 = (masked_image_float * 255).astype(np.uint8)

        return masked_image_uint8
    
    @staticmethod
    def cropSquare(image):
        """
        Crops the image from the center to make it square (1:1 ratio)
        Uses the smallest side as reference
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Square cropped image centered on the original
        """
        height, width = image.shape[:2]
        
        # The square side will be the smaller of the two dimensions
        square_side = min(width, height)
        
        # Calculate the starting point for the crop (centered)
        start_x = (width - square_side) // 2
        start_y = (height - square_side) // 2
        
        # Crop the image
        square_image = image[start_y:start_y + square_side, 
                            start_x:start_x + square_side]
        
        return square_image


class Viewer(QGraphicsView):
    # Initialization
    def __init__(self, parent=None) -> None:
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
    """
    def setImageFromPath(self, source_img_path):
        pixmap = QPixmap(source_img_path)
        self.setImageFromPixmap(pixmap)
    """
    def clearVariables(self) -> None:
        self.mask_item = None
        self.point_coordinates = []
        self.point_labels = []
        self.marker_items = []

    def setImageFromPixmap(self, pixmap: QPixmap) -> None:
        self.scene.clear()
        self.mask_item = None               # <-- Olvida la referencia a la máscara anterior.
        self.point_coordinates.clear()      # <-- Limpia la lista de coordenadas.
        self.point_labels.clear()           # <-- Limpia la lista de etiquetas.
        
        self.pixmap_item = QGraphicsPixmapItem(pixmap)        
        self.scene.addItem(self.pixmap_item)
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.scene_rect = self.scene.itemsBoundingRect()
    
    def addOverlay(self, pixmap: QPixmap) -> None:
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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
    
    def wheelEvent(self, event) -> None:
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
    
    def enterEvent(self, event) -> None:
        self.viewport().setCursor(Qt.ArrowCursor)
        super().enterEvent(event)
    
    def mouseMoveEvent(self, event) -> None:
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

    def mouseReleaseEvent(self, event) -> None:
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
    
    def mousePressEvent(self, event) -> None:
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

    def clearMask(self) -> None:
        if self.mask_item:
            self.scene.removeItem(self.mask_item)
            self.mask_item = None
    
    def clearAllPoints(self) -> None:
        if self.marker_items:
            for i, marker in enumerate(self.marker_items):
                self.scene.removeItem(marker)
            
            self.marker_items.clear()
            self.point_coordinates.clear()
            self.point_labels.clear()
        
    def showMask(self, show: bool) -> None:
        if self.mask_item:
            self.mask_item.setVisible(show)
    
    def showAllPoints(self, show: bool) -> None:
        if self.marker_items:
            for marker in self.marker_items:
                marker.setVisible(show)
    
    def clearLastPoint(self) -> None:
        if self.marker_items:
            last_point = self.marker_items.pop()
            self.scene.removeItem(last_point)
            self.point_coordinates.pop()
            self.point_labels.pop()
    
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

class MainWindow(QMainWindow):
    # Initialization
    def __init__(self) -> None:
        super().__init__()
        # Attributes
        FONT_FAMILY = 'Sans Serif'
        
        self.source_img_path = None
        self.processing = None
        self.doc = None
        self.calibration = Calibration()

        # Window Settings
        self.setWindowTitle("SACISMC")
        #self.setFixedWidth(1020)
        #self.setFixedHeight(720)
        #self.setFixedSize(1020, 720)  # Tamaño fijo de la ventana
        #self.setMinimumSize(1020, 720)
        #self.setMaximumSize(1020, 720)

        # Widgets
        self.viewer = Viewer()
        self.viewer.setFixedSize(720, 720)

        # Panel lateral derecho
        side_panel = self.createSidePanel()

        # Menu Bar
        menu_bar = self.menuBar()
        menu_file = menu_bar.addMenu("Archivo")
        action_open_img = menu_file.addAction("Abrir imagen")
        action_open_img.triggered.connect(self.openImage)
        submenu_extract = menu_file.addMenu("Exportar")
        action_extract_img = submenu_extract.addAction("Imagen original")
        action_extract_img.triggered.connect(self.saveImage)
        action_extract_filter_comparison = submenu_extract.addAction("Comparación filtro guiado")
        action_extract_filter_comparison.triggered.connect(self.exportFeatheredComparisonImage)
        action_extract_histogram = submenu_extract.addAction("Histograma de color")
        action_extract_histogram.triggered.connect(self.exportHistogram)


        menu_edit = menu_bar.addMenu("Editar")
        #menu_edit.setFont(self.menus_font)
        action_mask_color = menu_edit.addAction("Cambiar color máscara")
        action_mask_color.triggered.connect(self.openColorDialog)
        submenu_delete = menu_edit.addMenu("Eliminar")
        action_mask_color.triggered.connect(self.openColorDialog)
        action_delete_last_point = submenu_delete.addAction("Último punto")
        action_delete_last_point.triggered.connect(self.viewer.clearLastPoint)
        action_delete_all_points = submenu_delete.addAction("Todos los puntos")
        action_delete_all_points.triggered.connect(self.viewer.clearAllPoints)
        action_delete_mask = submenu_delete.addAction("Máscara")
        action_delete_mask.triggered.connect(self.viewer.clearMask)
        
        menu_calibrate = menu_bar.addMenu("Calibrar")
        action_select_checker = menu_calibrate.addAction("Seleccionar imagen Color Checker")
        action_select_checker.triggered.connect(self.selectColorCheckerimage)
        action_create_params = menu_calibrate.addAction("Crear parámetros de calibración")
        action_create_params.triggered.connect(self.createCalibrationParams)

        # Main Layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10) 
        main_layout.setSpacing(20)
        main_layout.addWidget(self.viewer)
        main_layout.addWidget(side_panel)
        main_layout.addStretch()

        # Containers and Layouts
        main_container = QWidget()
        main_container.setObjectName("MainContainer")
        main_container.setLayout(main_layout)
        self.setCentralWidget(main_container)

        # Styles
        styles = """
        QMainWindow {
            background-color: #2d2d2d;
        }
        QMenuBar {
            background-color: #2d2d2d;
            color: #e0e0e0;
            font-size: 14px;
            spacing: 0px;
            padding: 0px;
            margin: 10px 10px 10px 10px;
            border-bottom: 1px solid #707070;
        }
        QMenuBar::item {
            padding: 5px 8px 5px 8px;
            margin: 0px;
        }
        QMenuBar::item:selected {
            background: #424242;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }
        QMenu {
            font-size: 14px;
            background-color: #464646;
            color: #e0e0e0;
            padding: 6px 0px;
        }
        QMenu::item {
            padding: 4px 8px 5px 24px;
            margin: 0px 6px;
        }
        QMenu::item:selected {
            background-color: #308cc6;
            color: white;
            border-radius: 10px;
        }
        QPushButton{
            font-weight: bold;
            font-size: 14px;
            padding: 5px;
            margin: 0px;
            border: 1px solid #707070;
            border-radius: 7px;
            background-color: #424242;
            color: #e0e0e0;
        }
        QPushButton:hover {
            background-color: #313131;
            border: 0px solid #707070;
        }
        QGraphicsView {
            background-color: #313131;
            border: 0px solid #bdbdbd;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #707070;
            border-radius: 7px;
            margin-top: 1ex;
            padding-top: 5px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
            color: #e0e0e0;
        }
        QLabel {
            color: #e0e0e0;
            font-size: 13px;
            font-weight: normal;
        }
        QCheckBox {
            color: #e0e0e0;
        }
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            border: 1px solid #e0e0e0;
            border-radius: 3px;
            background-color: #e0e0e0;/*606060*/
        }
        QCheckBox::indicator:checked {
            width: 14px;
            height: 14px;
            border: 1px solid #e0e0e0;
            border-radius: 3px;
            background-color: #2b2b2b;
        }
        QDialog {
            background-color: #373737;
        }
        """
        # Aplica la hoja de estilos a toda la ventana
        self.setStyleSheet(styles)

    # Methods
    def createSidePanel(self) -> QWidget:
        """Crea el panel lateral con controles"""
        panel = QWidget()
        panel.setFixedWidth(300)
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        
        # Botones
        self.group_box_button = QWidget()
        group_button_layout = QVBoxLayout()
        group_button_layout.setContentsMargins(0, 0, 0, 0) 
        group_button_layout.setSpacing(6)

        self.button_take_photo = QPushButton("Tomar Foto")
        self.button_take_photo.clicked.connect(self.takePhotoClicked)
        self.button_take_photo.setFixedHeight(50) #40
        group_button_layout.addWidget(self.button_take_photo)

        print(self.button_take_photo.styleSheet())

        self.button_run = QPushButton("Segmentar")
        self.button_run.clicked.connect(self.runSegmentation)
        self.button_run.setFixedHeight(50) #40
        group_button_layout.addWidget(self.button_run)

        self.group_box_button.setLayout(group_button_layout)
        layout.addWidget(self.group_box_button)

        # Otro separador
        #line = QFrame()
        #line.setFrameShape(QFrame.HLine)
        #line.setFrameShadow(QFrame.Sunken)
        #layout.addWidget(line)

        self.group_box_others = QWidget()
        group_others_layout = QVBoxLayout()
        group_others_layout.setContentsMargins(0, 0, 0, 0) 
        group_others_layout.setSpacing(10)

        group_box_1 = QGroupBox("Color")
        group_layout_1 = QVBoxLayout()
        group_layout_1.setContentsMargins(13, 15, 13, 10) 
        group_layout_1.setSpacing(8)
        
        # Widget de color
        self.color_pantone = QLabel("PANTONE")
        self.color_pantone.setStyleSheet("font-weight: bold;")
        group_layout_1.addWidget(self.color_pantone)

        self.color_lab = QLabel("LAB")
        #self.color_lab.setStyleSheet("font-weight: normal;")
        group_layout_1.addWidget(self.color_lab)

        self.color_delta = QLabel("DELTA E")
        #self.color_delta.setStyleSheet("font-weight: normal;")
        group_layout_1.addWidget(self.color_delta)
        
        self.color_display = QFrame()
        self.color_display.setFixedHeight(80)
        self.color_display.setFrameStyle(QFrame.Box)
        #self.color_display.setStyleSheet("background-color: rgb(128, 128, 128); border: none;")
        group_layout_1.addWidget(self.color_display)
        
        # Label con valores RGB
        self.rgb_label = QLabel("Representación RGB")
        self.rgb_label.setStyleSheet("font-size: 11px;")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        group_layout_1.addWidget(self.rgb_label)
        
        group_box_1.setLayout(group_layout_1)
        group_others_layout.addWidget(group_box_1)

        group_box_2 = QGroupBox("Visualización")
        group_layout_2 = QVBoxLayout()
        group_layout_2.setContentsMargins(13, 15, 13, 10) 
        group_layout_2.setSpacing(7)
        
        # Checkbox
        self.checkbox_hide_mask = QCheckBox("Ocultar máscara")
        self.checkbox_hide_mask.setChecked(False)
        self.checkbox_hide_mask.stateChanged.connect(self.hideMask)
        group_layout_2.addWidget(self.checkbox_hide_mask)

        self.checkbox_hide_points = QCheckBox("Ocultar puntos")
        self.checkbox_hide_points.setChecked(False)
        self.checkbox_hide_points.stateChanged.connect(self.hidePoints)
        group_layout_2.addWidget(self.checkbox_hide_points)

        group_box_2.setLayout(group_layout_2)
        group_others_layout.addWidget(group_box_2)

        group_box_3 = QGroupBox("Registro")
        group_layout_3 = QVBoxLayout()
        group_layout_3.setContentsMargins(13, 15, 13, 10) 
        group_layout_3.setSpacing(10)

        self.input_id = QLineEdit()
        self.input_id.setPlaceholderText("Identificador")
        self.input_id.setFixedHeight(30)
        group_layout_3.addWidget(self.input_id)

        self.log_button = QPushButton("Guardar")
        self.log_button.clicked.connect(self.saveExcel)
        self.log_button.setFixedHeight(70) #40
        self.log_button.setStyleSheet("font-size: 16px;")
        group_layout_3.addWidget(self.log_button)

        group_box_3.setLayout(group_layout_3)
        group_others_layout.addWidget(group_box_3)


        self.group_box_others.setLayout(group_others_layout)
        self.group_box_others.hide()

        layout.addWidget(self.group_box_others)
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def takePhotoClicked(self) -> None:
        print("Botón presionado")
    
    def hideMask(self, state) -> None:
        if state == 2:
            self.viewer.showMask(False)
        else:
            self.viewer.showMask(True)

    def hidePoints(self, state) -> None:
        if state == 2:
            self.viewer.showAllPoints(False)
        else:
            self.viewer.showAllPoints(True)
    
    def updateColorDisplay(self, color_rgb: dict) -> None:
        """Actualiza el widget de visualización de color"""
        if color_rgb is not None:
            r = color_rgb["Pantone RGB"][0]
            g = color_rgb["Pantone RGB"][1]
            b = color_rgb["Pantone RGB"][2]
            self.color_display.setStyleSheet(
                f"background-color: rgb({r}, {g}, {b}); border: 2px solid #999;"
            )
            #self.rgb_label.setText(f"R: {r}  G: {g}  B: {b}")
            self.color_pantone.setText(color_rgb["Pantone Name"])
            self.color_lab.setText(f"LAB: ({color_rgb["Pantone LAB"][0]:.5f}, {color_rgb["Pantone LAB"][1]:.1f}, {color_rgb["Pantone LAB"][2]:.1f})")
            self.color_delta.setText(f"ΔE00: {color_rgb["Delta E"]:.5f}")
            self.checkbox_hide_mask.setChecked(False)
            self.checkbox_hide_points.setChecked(False)
    
    def saveCSV(self) -> None:
        id = self.input_id.text().strip()
        if not id:
            QMessageBox.warning(self, "Campo vacío", "Por favor ingrese un identificador antes de guardar.")
            return None
        
        main_color = self.processing.getMainColor()

        color_pantone: str = f"{main_color["Pantone Name"]}"
        color_lab_pantone: str = f"({main_color["Pantone LAB"][0]:.5f}, {main_color["Pantone LAB"][1]:.1f}, {main_color["Pantone LAB"][2]:.1f})"
        color_lab_original: str = f"({main_color["Original LAB"][0]:.5f}, {main_color["Original LAB"][1]:.1f}, {main_color["Original LAB"][2]:.1f})"
        delta_E: str = f"{main_color["Delta E"]:.5f}"
        img_path: str = f"{os.getcwd()}/registered_images/{id.lower().replace(" ","_")}.png"
        
        path_csv = "registros.xlsx"
        file_exists: bool = os.path.exists(path_csv)

        try:
            # Abrir el archivo en modo append
            with open(path_csv, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                # Si el archivo no existe, escribir los headers
                if not file_exists:
                    writer.writerow(['Identificador', 'Color PANTONE', 'Color LAB PANTONE', 'Color LAB Original', 
                                'ΔE00', 'Ubicación Imagen'])
                
                # Escribir los datos
                writer.writerow([id, color_pantone, color_lab_pantone, color_lab_original, 
                            delta_E, img_path])
            
            # Mostrar mensaje de éxito
            QMessageBox.information(self, "Éxito", 
                                f"Datos guardados correctamente en {path_csv}")
            
            # Limpiar el campo de texto después de guardar
            self.input_id.clear()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                            f"Error al guardar los datos: {str(e)}")

    def saveExcel(self) -> None:
        id = self.input_id.text().strip()
        if not id:
            QMessageBox.warning(self, "Campo vacío", "Por favor ingrese un identificador antes de guardar.")
            return None
        
        main_color = self.processing.getMainColor()

        color_pantone: str = f"{main_color["Pantone Name"]}"
        color_lab_pantone: str = f"({main_color["Pantone LAB"][0]:.5f}, {main_color["Pantone LAB"][1]:.1f}, {main_color["Pantone LAB"][2]:.1f})"
        color_lab_original: str = f"({main_color["Original LAB"][0]:.5f}, {main_color["Original LAB"][1]:.1f}, {main_color["Original LAB"][2]:.1f})"
        delta_E: str = f"{main_color["Delta E"]:.5f}"
        img_path: str = f"{os.getcwd()}/registered_images/{id.lower().replace(" ","_")}.png"
        
        path_excel = "registros.xlsx"

        new_data: dict = {
                            'Identificador': [id],
                            'Color PANTONE': [color_pantone],
                            'Color LAB PANTONE': [color_lab_pantone],
                            'Color LAB Original': [color_lab_original],
                            'ΔE00': [delta_E],
                            'Ubicación Imagen': [img_path]
                        }

        new_df = pd.DataFrame(new_data)

        try:
            if os.path.exists(path_excel):
                # Leer el archivo existente
                existing_df = pd.read_excel(path_excel)
                
                # Concatenar los datos existentes con los nuevos
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                # Si no existe, usar solo los nuevos datos
                final_df = new_df
            
            # Guardar el DataFrame en Excel
            final_df.to_excel(path_excel, index=False, engine='openpyxl')

            self.processing.saveOriginalImage(img_path)

            # Mostrar mensaje de éxito
            QMessageBox.information(self, "Éxito", 
                                f"Datos guardados correctamente en {path_excel}")
            
            # Limpiar el campo de texto después de guardar
            self.input_id.clear()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                            f"Error al guardar los datos: {str(e)}")

    
    def createCalibrationParams(self) -> None:
        color_patches = self.calibration.detectColorChecker()
        if color_patches is not None:
            self.calibration.saveCalibrationParams(color_patches)
            QMessageBox.information(self, 
                                    "Parametros de calibracion creados correctamente", 
                                    f"Guardados como {self.calibration.getParamsPath()}")

    def applyCalibrationParams(self) -> None:
        print("entra apply")

    def saveImage(self) -> None:
        if self.source_img_path is None:
            QMessageBox.warning(self, "Sin imagen", 
                            "No hay ninguna imagen cargada para guardar.")
            return None

        file_path, _ = QFileDialog.getSaveFileName(
                                    self,
                                    "Guardar imagen como",
                                    "imagen_sin_titulo.png",                 # Nombre sugerido
                                    "Imágenes PNG (*.png);;Imágenes JPEG (*.jpg *.jpeg);;Todos los archivos (*.*)"
                                )

        if not file_path:
            return None

        try:
            self.processing.saveOriginalImage(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                            f"Error al guardar la imagen: {str(e)}")
    
    def selectColorCheckerimage(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "resources/calibration", "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.calibration.setColorCheckerPath(path)
            QMessageBox.information(self, 
                                    "Nuevo Color Checker seleccionado", 
                                    f"Crea nuevamente los parámetros de calibración.")


    def openImage(self) -> None:
        self.source_img_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "resources/test_images", "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp)"
        )
        if self.source_img_path:
            self.processing = ImageProcessing(self.source_img_path, self.calibration)
            self.viewer.clearVariables()
            self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(self.processing.getScaledImage()))            
            self.doc = None
            if self.group_box_others.isVisible():
                self.group_box_others.hide()

    def runSegmentation(self) -> None:
        if self.source_img_path:
            if self.viewer.point_coordinates:
                self.processing.setInputPointArray(self.viewer.point_coordinates)
                self.processing.setInputLabelArray(self.viewer.point_labels)
                self.processing.setRawMask()
                self.processing.guidedFilter(self.processing.getFilterR(),
                                                self.processing.getFilterEPS())
                f_color = self.processing.getFeatheredMaskColor()
                feathered_mask_colored = self.processing.createColoredMask(self.processing.getScaledFeatheredMask(), f_color)
                self.viewer.addOverlay(self.viewer.fromCV2ToQPixmap(feathered_mask_colored))

                if not self.group_box_others.isVisible():
                    self.group_box_others.show()

                self.updateColorDisplay(self.processing.getColorByWeightedMedian())
                self.doc = Documentation()
            else:
                QMessageBox.warning(self, 
                                "Prompts no encontrados", 
                                "Por favor, crea los puntos antes de correr la segmentación.")
                if self.group_box_others.isVisible():
                    self.group_box_others.hide()

                if self.viewer.mask_item:
                    self.viewer.scene.removeItem(self.viewer.mask_item)
                    self.viewer.mask_item = None
        else:
            QMessageBox.warning(self, 
                                "Imagen no encontrada", 
                                "Por favor, carga una imagen antes de correr la segmentación.")

    def exportFeatheredComparisonImage(self) -> None:
        if self.doc:
            img_path = self.doc.createGuidedFilterComparisonImage(self.processing)
            QMessageBox.information(self, 
                                "Imagen exportada exitosamente", 
                                f"Guardada como {img_path}")
        else:
            QMessageBox.warning(self, 
                                "Mascara no encontrada", 
                                "Por favor, correr la segmentación antes de exportar la imagen.")

    def exportHistogram(self) -> None:
        if self.doc and self.processing:
            # 1. Obtener los datos del histograma desde ImageProcessing
            hist_data = self.processing.getSegmentedRegionHistogram()

            if hist_data:
                # 2. Generar y guardar la imagen del gráfico con Documentation
                img_path = self.doc.createHistogramImage(hist_data)
                QMessageBox.information(self, 
                                        "Histograma Exportado", 
                                        f"El histograma se ha guardado como {img_path}")
            else:
                QMessageBox.warning(self, 
                                    "Error", 
                                    "No se pudieron calcular los datos del histograma.")
        else:
            QMessageBox.warning(self, 
                                "Máscara no encontrada", 
                                "Por favor, corre la segmentación antes de exportar el histograma.")
    
    def openColorDialog(self) -> None:
        if self.processing:
            r,g,b = self.processing.getFeatheredMaskColor().tolist()
            default_color = QColor.fromRgb(r,g,b)
            new_color = QColorDialog.getColor(default_color, self, "Elige un color")
            if new_color.isValid():
                r = new_color.red()
                g = new_color.green()
                b = new_color.blue()
                self.processing.setFeatheredMaskColor([r,g,b])
        else:
            QMessageBox.warning(self, 
                                "Imagen no encontrada", 
                                "Por favor, carga una imagen antes de seleccionar el color.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec() #Start the event loop
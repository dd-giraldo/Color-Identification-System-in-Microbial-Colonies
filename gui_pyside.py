import sys
import os
from PySide6.QtCore import Qt
from PySide6.QtGui import (QPixmap, QImage, QPainter, QColor, QPalette)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QGraphicsScene, 
                                QGraphicsView, QFileDialog, QColorDialog, QGraphicsPixmapItem, QGraphicsEllipseItem, 
                                QPushButton, QMessageBox, QFrame, QCheckBox, QGroupBox, QLabel, QLineEdit, QTabWidget,
                                QSpinBox, QDoubleSpinBox, QFormLayout, QDialog, QComboBox)

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
        self.default_color_checker_img_path = "resources/calibration/color_checker.jpg"
        self.params_path = "resources/calibration/calibration_params.pickle"

    def getParamsPath(self):
        return self.params_path
    
    def setColorCheckerPath(self, path) -> None:
        self.default_color_checker_img_path = path
            
    def detectColorChecker(self):
        image = cv2.imread(self.default_color_checker_img_path)
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
    def __init__(self) -> None:
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
        self.top_colors = {}
        self.scaled_image_size = 720
        self.pantone_lab_colors = None
        self.pantone_name_colors = None
        self.original_image = None

        # Load PANTONE database
        self.pantone_database_path="resources/pantone_databases/PANTONE Solid Coated-V4.json"
        self.loadPantoneDatabase()

        # Create SAM2 predictor
        device = torch.device("cpu")
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        

    # Methods
    def loadImage(self, source_image_path, calibration) -> None:
        # Load image
        self.original_image = cv2.cvtColor(self.cropSquare(cv2.imread(source_image_path)), cv2.COLOR_BGR2RGB)
        model = calibration.reconstructModelFromParams()
        self.original_image = calibration.applyColorCorrection(self.original_image, model)
        self.original_image_size = self.original_image.shape[:2]
        self.scaled_image = self.decimateImage(self.original_image)
        self.predictor.set_image(self.scaled_image)

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
        if not is_mask:
            method = cv2.INTER_AREA
        else:
            method = cv2.INTER_NEAREST

        return cv2.resize(image, (self.scaled_image_size, self.scaled_image_size), interpolation=method)
    
    def interpolateImage(self, image: np.ndarray, is_mask: bool = False) -> np.ndarray:
        if not is_mask:
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
        print(np.max(self.feathered_mask))
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
        sorted_3_indx = np.argsort(delta_Es)[:3]

        for i, indx in enumerate(sorted_3_indx):
            name = self.pantone_name_colors[indx]
            lab = self.pantone_lab_colors[indx]
            rgb = np.astype(np.round(lab2rgb(lab)*255), np.uint8)
            self.top_colors[f"Color {i+1}"] = {
                                            "Pantone Name": name,
                                            "LAB": lab,
                                            "RGB": rgb,
                                            "ΔE00": delta_Es[indx]
                                            }
    
    def getTopColors(self) -> dict:
        return self.top_colors
    
    def getMainColor(self) -> dict:
        return self.main_color

    def estimateColorsByWeightedMedian(self, min_weight_threshold: float = 0.0) -> None:
        if self.original_image.size <= 0:
            return None
        # Convertir a LAB en rangos estándar
        image_lab = self.fromRGBtoLAB(self.original_image)

        # Filtrar pixels con peso significativo (> 0 para incluir toda la gradación)
        valid_mask: bool = self.feathered_mask > min_weight_threshold
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
        
        self.findNearestPantone(dominant_color_lab)

    def estimateColorByKMeans(n_clusters: int = 3,
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

    def estimateColorBySoftVoting(sigma: float = 10.0,
                                max_samples: int = 5000,
                                min_weight_threshold: float = 0.1) -> np.ndarray:
        """
        Método 3: Soft Voting Probabilístico (Optimizado Vectorizado)
        Para cada pixel, calcula probabilidades de pertenencia a cada Pantone usando
        CIEDE2000 y una función de kernel RBF. Acumula votos ponderados por la máscara
        emplumada y selecciona el Pantone con mayor score.
        
        OPTIMIZACIÓN: Usa cálculo vectorizado de CIEDE2000 para todos los pixels
        y Pantones simultáneamente, logrando una mejora de 100-1000x en velocidad.
        
        Args:
            image_rgb: Imagen original en RGB (height, width, 3), dtype uint8
            feathered_mask: Máscara emplumada en escala de grises [0, 1], dtype float32
            pantone_db_path: Ruta al archivo JSON con la base de datos Pantone
            sigma: Parámetro de temperatura para la función RBF (default: 10.0)
            max_samples: Número máximo de pixels a muestrear (para eficiencia)
            min_weight_threshold: Umbral mínimo de peso en máscara
        
        Returns:
            dominant_color_lab: Color predominante en CIELAB (del Pantone ganador), dtype float32
        """
        # Cargar base de datos Pantone
        self.loadPantoneDatabase()
        
        # Convertir a LAB en rangos estándar
        image_lab = self.fromRGBtoLAB(self.original_image)
        
        # Filtrar pixels con peso significativo
        valid_mask: bool = self.feathered_mask > min_weight_threshold
        pixels_lab = image_lab[valid_mask]
        weights = self.feathered_mask[valid_mask]
        
        # Submuestreo probabilístico si hay muchos pixels
        if len(pixels_lab) > max_samples:
            probabilities = weights / np.sum(weights)
            indx = np.random.choice(len(pixels_lab), max_samples, 
                                    replace=False, p=probabilities)
            pixels_lab = pixels_lab[indx]
            weights = weights[indx]
        
        # Extraer todos los colores LAB de Pantone en un array
        n_pantones: int = len(self.pantone_lab_colors)
        n_pixels: int = len(pixels_lab)
        
        # VECTORIZACIÓN: Calcular todas las distancias de una vez
        # Reshape para broadcasting: pixels (n_pixels, 1, 3) vs pantones (1, n_pantones, 3)
        pixels_reshaped = pixels_lab.reshape(n_pixels, 1, 3)
        pantones_reshaped = self.pantone_lab_colors.reshape(1, n_pantones, 3)
        
        # Calcular matriz de distancias (n_pixels x n_pantones) de forma vectorizada
        # deltaE_ciede2000 espera imágenes, así que agregamos dimensión espacial dummy
        pixels_broadcast = np.repeat(pixels_reshaped[:, np.newaxis, :, :], n_pantones, axis=1)
        pantones_broadcast = np.repeat(pantones_reshaped[np.newaxis, :, :, :], n_pixels, axis=0)
        
        # Calcular distancias vectorizadas (shape: n_pixels x n_pantones)
        distances = deltaE_ciede2000(pixels_broadcast, pantones_broadcast).squeeze()
        
        # Convertir distancias a probabilidades usando RBF kernel
        # Aplicar sobre toda la matriz de distancias
        probabilities = np.exp(-distances**2 / (2 * sigma**2))
        
        # Normalizar probabilidades por fila (cada pixel suma 1.0)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        # Acumular scores ponderados por peso espacial
        # weights shape: (n_pixels,) -> reshape a (n_pixels, 1) para broadcasting
        weighted_probs = probabilities * weights.reshape(-1, 1)
        scores = weighted_probs.sum(axis=0)  # Sumar sobre pixels
        
        # Seleccionar Pantone con mayor score
        winner_idx = np.argmax(scores)
        dominant_color_lab = np.array(pantone_db[winner_idx]['lab'], dtype=np.float32)
        
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
        # Solo nos interesa el clic izquierdo p ara iniciar la lógica
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


class ConfigManager:
    """Gestor de configuración con soporte para JSON"""
    
    def __init__(self) -> None:
        self.default_config_path = "resources/config/default_config.json"
        self.user_config_path = "resources/config/user_config.json"
        self.style_path = "resources/config/dark_theme.qss"
        self.config = None
        self.style = None

        # Cargar configuración
        self.loadConfig()
        self.loadStyle()
    
    def loadConfig(self):
        """Carga la configuración (usuario si existe, sino default)"""
        # Intentar cargar configuración de usuario primero
        if os.path.exists(self.user_config_path):
            try:
                with open(self.user_config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print("Configuración de usuario cargada")
                return
            except Exception as e:
                print(f"Error al cargar configuración de usuario: {e}")
        
        # Si no existe o falla, cargar configuración por defecto
        self.loadDefaultConfig()
    
    def loadDefaultConfig(self) -> None:
        """Carga la configuración por defecto"""
        with open(self.default_config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def saveUserConfig(self, config_dict):
        """Guarda la configuración del usuario"""
        self.config = config_dict
        with open(self.user_config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        print(f"Configuración guardada en {self.user_config_path}")
    
    def getConfig(self):
        """Retorna la configuración actual"""
        return self.config
    
    def getValue(self, category, key):
        """Obtiene un valor específico de la configuración"""
        return self.config.get(category, {}).get(key)
    
    def resetToDefault(self):
        """Resetea a la configuración por defecto y la guarda"""
        self.loadDefaultConfig()
        # Eliminar configuración de usuario
        if os.path.exists(self.user_config_path):
            os.remove(self.user_config_path)
        print("Configuración reseteada a valores por defecto")
    
    def loadStyle(self):
        """Carga el archivo de estilos"""
        try:
            with open(self.style_path, 'r', encoding='utf-8') as f:
                self.style = f.read()
        except Exception as e:
            print(f"Error al cargar estilos: {e}")
    
    def getStyle(self):
        """Retorna el estilo cargado"""
        return self.style


class ConfigDialog(QDialog):
    """Ventana de configuración con pestañas para diferentes parámetros"""
    
    def __init__(self, parent=None, config_manager=None):
        super().__init__(parent)
        self.config_manager = config_manager
        
        # Configuración de la ventana
        self.setWindowTitle("Configuración")
        self.setMinimumSize(500, 400)

        flags = self.windowFlags()

        self.setWindowFlags(
            Qt.Dialog |                    # Es un diálogo
            Qt.WindowCloseButtonHint |     # Botón de cerrar visible
            Qt.WindowTitleHint |           # Barra de título visible
            Qt.WindowSystemMenuHint        # Menú del sistema visible
        )

        self.setModal(True)  # Hace que sea modal (bloquea la ventana principal)

        self.setAttribute(Qt.WA_DeleteOnClose, False)
        
        # Layout principal
        main_layout = QVBoxLayout()
        
        # Crear el widget de pestañas
        self.tab_widget = QTabWidget()
        
        # Crear las diferentes pestañas
        self.tab_widget.addTab(self.createCameraTab(), "Camera")
        self.tab_widget.addTab(self.createCalibrationTab(), "Calibration")
        self.tab_widget.addTab(self.createSegmentationTab(), "Segmentation")
        self.tab_widget.addTab(self.createColorTab(), "Color")
        self.tab_widget.addTab(self.createExportTab(), "Export")
        
        main_layout.addWidget(self.tab_widget)
        
        # Botones de acción
        button_layout = self.createButtonLayout()
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        # Cargar valores desde el config manager
        if self.config_manager:
            self.loadConfigValues()
    
    def closeEvent(self, event):
        """Maneja el evento de cierre de la ventana"""
        # Permitir cerrar la ventana siempre
        event.accept()
    
    def keyPressEvent(self, event):
        """Maneja eventos de teclado"""
        # Permitir cerrar con ESC
        if event.key() == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)
    
    def reject(self):
        """Sobrescribe reject para asegurar que el diálogo se cierre"""
        try:
            super().reject()
        except Exception as e:
            print(f"Error al cerrar diálogo: {e}")
        finally:
            self.close()
    
    def accept(self):
        """Sobrescribe accept para asegurar que el diálogo se cierre"""
        try:
            super().accept()
        except Exception as e:
            print(f"Error al aceptar diálogo: {e}")
        finally:
            self.close()
    
    # ==================== TAB 1: CAMERA ====================
    def createCameraTab(self):
        """Pestaña para configurar parámetros de la cámara"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Group Box sin título
        camera_group = QGroupBox()
        camera_layout = QFormLayout()
        
        # Auto gain (CheckBox)
        self.check_auto_gain = QCheckBox()
        self.check_auto_gain.setChecked(True)
        camera_layout.addRow("Auto gain:", self.check_auto_gain)
        
        # Preset gain (ComboBox)
        self.combo_preset_gain = QComboBox()
        self.combo_preset_gain.addItems(["Low", "Medium", "High"])
        camera_layout.addRow("Preset gain:", self.combo_preset_gain)
        
        # Exposition time (SpinBox en microsegundos)
        self.spin_exposition_time = QSpinBox()
        self.spin_exposition_time.setRange(100, 1000000)  # 100µs a 1s
        self.spin_exposition_time.setValue(10000)
        self.spin_exposition_time.setSuffix(" µs")
        self.spin_exposition_time.setSingleStep(1000)
        camera_layout.addRow("Exposition time:", self.spin_exposition_time)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    # ==================== TAB 2: CALIBRATION ====================
    def createCalibrationTab(self):
        """Pestaña para configurar parámetros de calibración"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Group Box sin título
        calibration_group = QGroupBox()
        calibration_layout = QVBoxLayout()
        calibration_layout.setSpacing(15)
        
        # Color Checker image path
        checker_layout = QHBoxLayout()
        btn_color_checker = QPushButton("Seleccionar Color Checker")
        btn_color_checker.clicked.connect(self.selectColorCheckerPath)
        self.label_color_checker_path = QLabel("No seleccionado")
        self.label_color_checker_path.setWordWrap(True)
        self.label_color_checker_path.setStyleSheet("font-size: 11px; color: #a0a0a0;")
        checker_layout.addWidget(btn_color_checker)
        checker_layout.addWidget(self.label_color_checker_path, 1)
        
        calibration_layout.addWidget(QLabel("Color Checker image path:"))
        calibration_layout.addLayout(checker_layout)
        
        # Calibration params file path
        params_layout = QHBoxLayout()
        btn_calib_params = QPushButton("Seleccionar Parámetros")
        btn_calib_params.clicked.connect(self.selectCalibParamsPath)
        self.label_calib_params_path = QLabel("No seleccionado")
        self.label_calib_params_path.setWordWrap(True)
        self.label_calib_params_path.setStyleSheet("font-size: 11px; color: #a0a0a0;")
        params_layout.addWidget(btn_calib_params)
        params_layout.addWidget(self.label_calib_params_path, 1)
        
        calibration_layout.addWidget(QLabel("Calibration params file path:"))
        calibration_layout.addLayout(params_layout)
        
        calibration_group.setLayout(calibration_layout)
        layout.addWidget(calibration_group)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def selectColorCheckerPath(self):
        """Abre diálogo para seleccionar imagen de Color Checker"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Color Checker",
            "resources/calibration",
            "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.label_color_checker_path.setText(path)
    
    def selectCalibParamsPath(self):
        """Abre diálogo para seleccionar archivo de parámetros"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Parámetros de Calibración",
            "resources/calibration",
            "Pickle Files (*.pickle *.pkl);;Todos los archivos (*.*)"
        )
        if path:
            self.label_calib_params_path.setText(path)
    
    # ==================== TAB 3: SEGMENTATION ====================
    def createSegmentationTab(self):
        """Pestaña para configurar parámetros de segmentación"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Group Box "Scene"
        scene_group = QGroupBox("Scene")
        scene_layout = QFormLayout()
        
        # Image scale size
        self.spin_image_scale = QSpinBox()
        self.spin_image_scale.setRange(256, 1920)
        self.spin_image_scale.setValue(720)
        self.spin_image_scale.setSingleStep(64)
        self.spin_image_scale.setSuffix(" px")
        scene_layout.addRow("Image scale size:", self.spin_image_scale)
        
        # Marker radius
        self.spin_marker_radius = QSpinBox()
        self.spin_marker_radius.setRange(1, 20)
        self.spin_marker_radius.setValue(1)
        self.spin_marker_radius.setSuffix(" px")
        scene_layout.addRow("Marker radius:", self.spin_marker_radius)
        
        scene_group.setLayout(scene_layout)
        layout.addWidget(scene_group)
        
        # Group Box "Mask"
        mask_group = QGroupBox("Mask")
        mask_layout = QFormLayout()
        
        # Mask color (Frame + Button)
        mask_color_layout = QHBoxLayout()
        self.frame_mask_color = QFrame()
        self.frame_mask_color.setFixedSize(60, 30)
        self.frame_mask_color.setFrameStyle(QFrame.Box)
        self.frame_mask_color.setStyleSheet("background-color: rgb(158, 16, 127); border: 2px solid #999;")
        
        btn_mask_color = QPushButton("Cambiar color")
        btn_mask_color.clicked.connect(self.selectMaskColor)
        
        mask_color_layout.addWidget(self.frame_mask_color)
        mask_color_layout.addWidget(btn_mask_color)
        mask_color_layout.addStretch()
        
        mask_layout.addRow("Mask color:", mask_color_layout)
        
        mask_group.setLayout(mask_layout)
        layout.addWidget(mask_group)
        
        # Group Box "Guided Filter"
        filter_group = QGroupBox("Guided Filter")
        filter_layout = QFormLayout()
        
        # Radius
        self.spin_filter_radius = QSpinBox()
        self.spin_filter_radius.setRange(1, 50)
        self.spin_filter_radius.setValue(2)
        self.spin_filter_radius.setSuffix(" px")
        filter_layout.addRow("Radius:", self.spin_filter_radius)
        
        # Epsilon
        self.spin_filter_epsilon = QDoubleSpinBox()
        self.spin_filter_epsilon.setRange(0.01, 1.0)
        self.spin_filter_epsilon.setValue(0.3)
        self.spin_filter_epsilon.setDecimals(2)
        self.spin_filter_epsilon.setSingleStep(0.01)
        self.spin_filter_epsilon.setSuffix("²")
        filter_layout.addRow("Epsilon (ε²):", self.spin_filter_epsilon)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def selectMaskColor(self):
        """Abre diálogo para seleccionar color de máscara"""
        # Obtener color actual del frame
        current_style = self.frame_mask_color.styleSheet()
        # Extraer RGB del estilo (formato: "background-color: rgb(r, g, b);")
        import re
        match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', current_style)
        if match:
            r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
            current_color = QColor(r, g, b)
        else:
            current_color = QColor(158, 16, 127)
        
        # Abrir diálogo de color
        new_color = QColorDialog.getColor(current_color, self, "Seleccionar color de máscara")
        
        if new_color.isValid():
            r, g, b = new_color.red(), new_color.green(), new_color.blue()
            self.frame_mask_color.setStyleSheet(
                f"background-color: rgb({r}, {g}, {b}); border: 2px solid #999;"
            )
    
    # ==================== TAB 4: COLOR ====================
    def createColorTab(self):
        """Pestaña para configurar parámetros de color"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Group Box "Pantone"
        pantone_group = QGroupBox("Pantone")
        pantone_layout = QVBoxLayout()
        pantone_layout.setSpacing(10)
        
        # Pantone Database File
        pantone_file_layout = QHBoxLayout()
        btn_pantone_file = QPushButton("Seleccionar Database")
        btn_pantone_file.clicked.connect(self.selectPantoneDatabase)
        self.label_pantone_path = QLabel("No seleccionado")
        self.label_pantone_path.setWordWrap(True)
        self.label_pantone_path.setStyleSheet("font-size: 11px; color: #a0a0a0;")
        pantone_file_layout.addWidget(btn_pantone_file)
        pantone_file_layout.addWidget(self.label_pantone_path, 1)
        
        pantone_layout.addWidget(QLabel("Pantone Database File:"))
        pantone_layout.addLayout(pantone_file_layout)
        
        pantone_group.setLayout(pantone_layout)
        layout.addWidget(pantone_group)
        
        # Group Box "Color Method"
        method_group = QGroupBox("Color Method")
        method_layout = QVBoxLayout()
        
        # Select method (ComboBox)
        method_select_layout = QFormLayout()
        self.combo_color_method = QComboBox()
        self.combo_color_method.addItems(["Median", "KMeans", "SoftVoting"])
        self.combo_color_method.currentTextChanged.connect(self.onColorMethodChanged)
        method_select_layout.addRow("Select method:", self.combo_color_method)
        method_layout.addLayout(method_select_layout)
        
        # Contenedor para parámetros condicionales
        self.method_params_layout = QFormLayout()
        
        # Parámetros para Median
        self.label_threshold_median = QLabel("Threshold Median:")
        self.spin_threshold_median = QDoubleSpinBox()
        self.spin_threshold_median.setRange(0.0, 1.0)
        self.spin_threshold_median.setValue(0.0)
        self.spin_threshold_median.setDecimals(2)
        self.spin_threshold_median.setSingleStep(0.05)
        self.method_params_layout.addRow(self.label_threshold_median, self.spin_threshold_median)
        
        # Parámetros para KMeans
        self.label_threshold_kmeans = QLabel("Threshold KMeans:")
        self.spin_threshold_kmeans = QDoubleSpinBox()
        self.spin_threshold_kmeans.setRange(0.0, 1.0)
        self.spin_threshold_kmeans.setValue(0.3)
        self.spin_threshold_kmeans.setDecimals(2)
        self.spin_threshold_kmeans.setSingleStep(0.05)
        self.method_params_layout.addRow(self.label_threshold_kmeans, self.spin_threshold_kmeans)
        
        self.label_number_clusters = QLabel("Number Clusters:")
        self.spin_number_clusters = QSpinBox()
        self.spin_number_clusters.setRange(2, 10)
        self.spin_number_clusters.setValue(3)
        self.method_params_layout.addRow(self.label_number_clusters, self.spin_number_clusters)
        
        # Parámetros para SoftVoting
        self.label_threshold_softvoting = QLabel("Threshold SoftVoting:")
        self.spin_threshold_softvoting = QDoubleSpinBox()
        self.spin_threshold_softvoting.setRange(0.0, 1.0)
        self.spin_threshold_softvoting.setValue(0.1)
        self.spin_threshold_softvoting.setDecimals(2)
        self.spin_threshold_softvoting.setSingleStep(0.05)
        self.method_params_layout.addRow(self.label_threshold_softvoting, self.spin_threshold_softvoting)
        
        self.label_sigma = QLabel("Sigma:")
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(1.0, 100.0)
        self.spin_sigma.setValue(10.0)
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setSingleStep(1.0)
        self.method_params_layout.addRow(self.label_sigma, self.spin_sigma)
        
        self.label_max_samples = QLabel("Max Samples:")
        self.spin_max_samples = QSpinBox()
        self.spin_max_samples.setRange(100, 50000)
        self.spin_max_samples.setValue(5000)
        self.spin_max_samples.setSingleStep(500)
        self.method_params_layout.addRow(self.label_max_samples, self.spin_max_samples)
        
        method_layout.addLayout(self.method_params_layout)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        
        # Mostrar solo los parámetros del método por defecto
        self.onColorMethodChanged("Median")
        
        return tab
    
    def selectPantoneDatabase(self):
        """Abre diálogo para seleccionar base de datos Pantone"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Base de Datos Pantone",
            "resources/pantone_databases",
            "JSON Files (*.json);;Todos los archivos (*.*)"
        )
        if path:
            self.label_pantone_path.setText(path)
    
    def onColorMethodChanged(self, method):
        """Muestra/oculta parámetros según el método seleccionado"""
        # Ocultar todos los parámetros primero
        self.label_threshold_median.hide()
        self.spin_threshold_median.hide()
        self.label_threshold_kmeans.hide()
        self.spin_threshold_kmeans.hide()
        self.label_number_clusters.hide()
        self.spin_number_clusters.hide()
        self.label_threshold_softvoting.hide()
        self.spin_threshold_softvoting.hide()
        self.label_sigma.hide()
        self.spin_sigma.hide()
        self.label_max_samples.hide()
        self.spin_max_samples.hide()
        
        # Mostrar solo los parámetros del método seleccionado
        if method == "Median":
            self.label_threshold_median.show()
            self.spin_threshold_median.show()
        elif method == "KMeans":
            self.label_threshold_kmeans.show()
            self.spin_threshold_kmeans.show()
            self.label_number_clusters.show()
            self.spin_number_clusters.show()
        elif method == "SoftVoting":
            self.label_threshold_softvoting.show()
            self.spin_threshold_softvoting.show()
            self.label_sigma.show()
            self.spin_sigma.show()
            self.label_max_samples.show()
            self.spin_max_samples.show()
    
    # ==================== TAB 5: EXPORT ====================
    def createExportTab(self):
        """Pestaña para configurar parámetros de exportación"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Group Box sin título
        export_group = QGroupBox()
        export_layout = QFormLayout()
        
        # Resolution DPI
        self.spin_export_dpi = QSpinBox()
        self.spin_export_dpi.setRange(72, 600)
        self.spin_export_dpi.setValue(300)
        self.spin_export_dpi.setSingleStep(50)
        self.spin_export_dpi.setSuffix(" dpi")
        export_layout.addRow("Resolution DPI:", self.spin_export_dpi)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    # ==================== BUTTONS ====================
    def createButtonLayout(self):
        """Crea los botones de Aceptar y Cancelar"""
        button_layout = QHBoxLayout()

        # Botón Restaurar por defecto (a la izquierda)
        btn_restore = QPushButton("Restaurar")
        btn_restore.clicked.connect(self.restoreDefaults)
        button_layout.addWidget(btn_restore)

        button_layout.addStretch()
        
        # Botón Cancelar
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)
        
        # Botón Aceptar
        btn_accept = QPushButton("Aceptar")
        btn_accept.clicked.connect(self.accept)
        #btn_accept.setDefault(True)
        button_layout.addWidget(btn_accept)
        
        return button_layout
    
    # ==================== LOAD/SAVE CONFIG ====================
    def loadConfigValues(self):
        """Carga los valores desde el config manager"""
        if not self.config_manager:
            return
        
        config = self.config_manager.getConfig()
        
        # Camera
        self.check_auto_gain.setChecked(config['camera']['auto_gain'])
        self.combo_preset_gain.setCurrentText(config['camera']['preset_gain'])
        self.spin_exposition_time.setValue(config['camera']['exposition_time'])
        
        # Calibration
        self.label_color_checker_path.setText(config['calibration']['color_checker_path'])
        self.label_calib_params_path.setText(config['calibration']['calibration_params_path'])
        
        # Segmentation
        self.spin_image_scale.setValue(config['segmentation']['scene']['image_scale_size'])
        self.spin_marker_radius.setValue(config['segmentation']['scene']['marker_radius'])
        
        mask_color = config['segmentation']['mask']['mask_color']
        self.frame_mask_color.setStyleSheet(
            f"background-color: rgb({mask_color[0]}, {mask_color[1]}, {mask_color[2]}); border: 2px solid #999;"
        )
        
        self.spin_filter_radius.setValue(config['segmentation']['guided_filter']['radius'])
        self.spin_filter_epsilon.setValue(config['segmentation']['guided_filter']['epsilon'])
        
        # Color
        self.label_pantone_path.setText(config['color']['pantone']['database_file'])
        self.combo_color_method.setCurrentText(config['color']['method']['selected_method'])
        self.spin_threshold_median.setValue(config['color']['method']['threshold_median'])
        self.spin_threshold_kmeans.setValue(config['color']['method']['threshold_kmeans'])
        self.spin_number_clusters.setValue(config['color']['method']['number_clusters'])
        self.spin_threshold_softvoting.setValue(config['color']['method']['threshold_softvoting'])
        self.spin_sigma.setValue(config['color']['method']['sigma'])
        self.spin_max_samples.setValue(config['color']['method']['max_samples'])
        
        # Export
        self.spin_export_dpi.setValue(config['export']['dpi'])
    
    def restoreDefaults(self):
        """Restaura los valores por defecto"""
        reply = QMessageBox.question(
            self,
            "Restaurar configuración por defecto",
            "¿Estás seguro de que deseas restaurar la configuración por defecto?\n"
            "Se perderán todos los cambios personalizados.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.config_manager:
                self.config_manager.resetToDefault()
                self.loadConfigValues()
                QMessageBox.information(
                    self,
                    "Configuración restaurada",
                    "La configuración ha sido restaurada a los valores por defecto."
                )
    
    def getValues(self):
        """Retorna un diccionario con todos los valores configurados"""
        # Extraer RGB del color de máscara
        import re
        mask_style = self.frame_mask_color.styleSheet()
        match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', mask_style)
        if match:
            mask_color = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
        else:
            mask_color = [158, 16, 127]

        dict_values = {
            'camera': {
                'auto_gain': self.check_auto_gain.isChecked(),
                'preset_gain': self.combo_preset_gain.currentText(),
                'exposition_time': self.spin_exposition_time.value(),
                'description': 'Parámetros de la cámara'
            },
            'calibration': {
                'color_checker_path': self.label_color_checker_path.text(),
                'calibration_params_path': self.label_calib_params_path.text(),
                'description': 'Rutas de calibración'
            },
            'segmentation': {
                'scene': {
                    'image_scale_size': self.spin_image_scale.value(),
                    'marker_radius': self.spin_marker_radius.value()
                },
                'mask': {
                    'mask_color': mask_color
                },
                'guided_filter': {
                    'radius': self.spin_filter_radius.value(),
                    'epsilon': self.spin_filter_epsilon.value()
                },
                'description': 'Parámetros de segmentación'
            },
            'color': {
                'pantone': {
                    'database_file': self.label_pantone_path.text()
                },
                'method': {
                    'selected_method': self.combo_color_method.currentText(),
                    'threshold_median': self.spin_threshold_median.value(),
                    'threshold_kmeans': self.spin_threshold_kmeans.value(),
                    'number_clusters': self.spin_number_clusters.value(),
                    'threshold_softvoting': self.spin_threshold_softvoting.value(),
                    'sigma': self.spin_sigma.value(),
                    'max_samples': self.spin_max_samples.value()
                },
                'description': 'Parámetros de estimación de color'
            },
            'export': {
                'dpi': self.spin_export_dpi.value(),
                'description': 'Parámetros de exportación'
            }
        }
        
        return dict_values


class MainWindow(QMainWindow):
    # Initialization
    def __init__(self) -> None:
        super().__init__()
        # Attributes
        FONT_FAMILY = 'Sans Serif'
        
        self.source_img_path = None
        self.doc = None

        self.calibration = Calibration()
        self.processing = ImageProcessing()

        # NUEVO: Inicializar el gestor de configuración
        self.config_manager = ConfigManager()

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

        action_settings = menu_file.addAction("Configuración")
        action_settings.triggered.connect(self.openSettingsDialog)
        #menu_file.addSeparator()

        menu_edit = menu_bar.addMenu("Editar")
        #menu_edit.setFont(self.menus_font)
        action_mask_color = menu_edit.addAction("Cambiar color máscara")
        action_mask_color.triggered.connect(self.openColorDialog)
        submenu_delete = menu_edit.addMenu("Eliminar")
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
        self.setStyleSheet(self.config_manager.getStyle())

        # Initial config
        #self.applyInitialConfig()
        self.applyConfigToComponents(self.config_manager.getConfig())

    # Methods
    def applyInitialConfig(self):
        """Aplica la configuración inicial a los componentes que lo necesitan"""
        config = self.config_manager.getConfig()
        
        # Aplicar configuración de visualización
        self.viewer.min_markpoint_radius = config['segmentation']['scene']['marker_radius']
        
        # Cargar rutas de calibración
        self.calibration.color_checker_img_path = config['calibration']['color_checker_path']
        self.calibration.params_path = config['calibration']['calibration_params_path']

    def openSettingsDialog(self) -> None:
        """Abre el diálogo de configuración"""
        dialog = ConfigDialog(self, self.config_manager)
        
        # Aplicar el mismo estilo que la ventana principal
        dialog.setStyleSheet(self.config_manager.getStyle())
        
        # Mostrar el diálogo y procesar resultado
        if dialog.exec() == QDialog.Accepted:
            # Obtener los valores configurados
            values = dialog.getValues()

            self.config_manager.saveUserConfig(values)
            
            # Aplicar los valores al processing si existe
            self.applyConfigToComponents(values)
                
            # Mostrar mensaje de confirmación
            QMessageBox.information(
                self, 
                "Configuración aplicada",
                "Los parámetros se han guardado y aplicado correctamente."
            )

    def applyConfigToComponents(self, config):
        """Aplica la configuración a todos los componentes de la aplicación"""
        # Aplicar a Calibration
        self.calibration.color_checker_img_path = config['calibration']['color_checker_path']
        self.calibration.params_path = config['calibration']['calibration_params_path']
        
        # Aplicar a Viewer
        self.viewer.min_markpoint_radius = config['segmentation']['scene']['marker_radius']
        
        # Aplicar a Processing si existe
        if self.processing:
            self.processing.scaled_image_size = config['segmentation']['scene']['image_scale_size']
            self.processing.r_filter = config['segmentation']['guided_filter']['radius']
            self.processing.eps_filter = config['segmentation']['guided_filter']['epsilon']**2
            self.processing.setFeatheredMaskColor(config['segmentation']['mask']['mask_color'])
            self.processing.pantone_database_path = config['color']['pantone']['database_file']
            
            # Actualizar tamaño de escala (requiere reiniciar processing con nueva imagen)
            # Este parámetro se aplicará en la próxima carga de imagen
        
        print("Configuración aplicada a todos los componentes")

    def createSidePanel(self) -> QWidget:
        """Crea el panel lateral con controles"""
        panel = QWidget()
        panel.setFixedWidth(400)
        
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

        # -------------------- DISPLAY --------------------

        group_box_1 = QGroupBox("Visualización")
        group_layout_1 = QVBoxLayout()
        group_layout_1.setContentsMargins(13, 15, 13, 10) 
        group_layout_1.setSpacing(7)
        
        # Checkbox
        self.checkbox_hide_mask = QCheckBox("Ocultar máscara")
        self.checkbox_hide_mask.setChecked(False)
        self.checkbox_hide_mask.stateChanged.connect(self.hideMask)
        group_layout_1.addWidget(self.checkbox_hide_mask)

        self.checkbox_hide_points = QCheckBox("Ocultar puntos")
        self.checkbox_hide_points.setChecked(False)
        self.checkbox_hide_points.stateChanged.connect(self.hidePoints)
        group_layout_1.addWidget(self.checkbox_hide_points)

        group_box_1.setLayout(group_layout_1)
        group_others_layout.addWidget(group_box_1)

        # -------------------- COLOR --------------------

        group_box_2 = QGroupBox("Color")
        group_layout_2 = QVBoxLayout()
        group_layout_2.setContentsMargins(13, 15, 13, 10) 
        group_layout_2.setSpacing(8)
        
        # Widget de color
        self.color_pantone = QLabel("PANTONE")
        self.color_pantone.setStyleSheet("font-weight: bold;")
        group_layout_2.addWidget(self.color_pantone)

        self.color_lab = QLabel("LAB")
        self.color_lab.setStyleSheet("font-weight: normal;")
        group_layout_2.addWidget(self.color_lab)

        self.color_delta = QLabel("DELTA E")
        #self.color_delta.setStyleSheet("font-weight: normal;")
        group_layout_2.addWidget(self.color_delta)
        
        self.color_display = QFrame()
        self.color_display.setFixedHeight(80)
        self.color_display.setFrameStyle(QFrame.Box)
        #self.color_display.setStyleSheet("background-color: rgb(128, 128, 128); border: none;")
        group_layout_2.addWidget(self.color_display)
        
        # Label con valores RGB
        self.rgb_label = QLabel("Representación RGB")
        self.rgb_label.setStyleSheet("font-size: 11px;")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        group_layout_2.addWidget(self.rgb_label)
        
        group_box_2.setLayout(group_layout_2)
        group_others_layout.addWidget(group_box_2)

        # -------------------- REGISTER --------------------

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
    
    def updateColorDisplay(self, colors: dict) -> None:
        """Actualiza el widget de visualización de color"""
        if colors is not None:
            r = colors["Color 1"]["RGB"][0]
            g = colors["Color 1"]["RGB"][1]
            b = colors["Color 1"]["RGB"][2]
            self.color_display.setStyleSheet(
                f"background-color: rgb({r}, {g}, {b}); border: 2px solid #999;"
            )
            #self.rgb_label.setText(f"R: {r}  G: {g}  B: {b}")
            self.color_pantone.setText(colors["Color 1"]["Pantone Name"])
            l = colors["Color 1"]["LAB"][0]
            a = colors["Color 1"]["LAB"][1]
            b = colors["Color 1"]["LAB"][2]
            self.color_lab.setText(f"LAB: ({l:.5f}, {a:.1f}, {b:.1f})")
            self.color_delta.setText(f"ΔE00: {colors["Color 1"]["ΔE00"]:.5f}")
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
            self.processing.loadImage(self.source_img_path, self.calibration)
            self.viewer.clearVariables()
            self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(self.processing.getScaledImage()))            
            self.doc = None
            if self.group_box_others.isVisible():
                self.group_box_others.hide()

    def runSegmentation(self) -> None:
        if self.source_img_path:
            if self.viewer.point_coordinates:
                # Obtener configuración de método de color
                config = self.config_manager.getConfig()
                color_method = config['color']['method']['selected_method']

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

                
                if color_method == "Median":
                    self.processing.estimateColorsByWeightedMedian(min_weight_threshold=config['color']['method']['threshold_median'] )
                elif color_method == "KMeans":
                    self.processing.estimateColorByKMeans(n_clusters=config['color']['method']['number_clusters'],
                                                        min_weight_threshold=config['color']['method']['threshold_kmeans'])
                elif color_method == "SoftVoting":
                    self.processing.estimateColorBySoftVoting(sigma=config['color']['method']['sigma'],
                                                            max_samples=config['color']['method']['max_samples'],
                                                            min_weight_threshold=config['color']['method']['threshold_softvoting'])
                
                self.updateColorDisplay(self.processing.getTopColors())
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
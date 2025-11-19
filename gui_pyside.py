import sys
import os
from datetime import datetime
from PySide6.QtCore import (Qt, QRectF)
from PySide6.QtGui import (QPixmap, QImage, QPainter, QColor, QPalette)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QGraphicsScene, 
                                QGraphicsView, QFileDialog, QColorDialog, QGraphicsPixmapItem, QGraphicsEllipseItem, 
                                QPushButton, QMessageBox, QFrame, QCheckBox, QRadioButton, QGroupBox, QLabel, QLineEdit, QTabWidget,
                                QSpinBox, QDoubleSpinBox, QFormLayout, QDialog, QComboBox, QStackedWidget)

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
    def __init__(self, config) -> None:
        self.params_path = "resources/calibration/default_calibration_params.pickle"
        self.config = config
        self.color_checker_raw_image = None
        self.color_patches = None
        self.img_draw = None

    def getParamsPath(self) -> str:
        return self.params_path
    
    def loadRawImage(self, file_path: str) -> None:
        ext: str = os.path.splitext(file_path)[1].lower()
        if ext == ".png":
            self.color_checker_raw_image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        elif ext == ".npy":
            self.color_checker_raw_image = np.load(file_path)
        else:
            QMessageBox.warning(self, "Formato no soportado",
                                f"El formato {ext} no es válido para abrir.")
        
    def getRawImage(self):
        return self.color_checker_raw_image

    def setDrawImage(self, img) -> None:
        self.img_draw = img

    def getDrawImage(self):
        return self.img_draw
            
    def detectColorChecker(self, drawPatches: bool = False) -> None:
        imageBGR = cv2.cvtColor(self.color_checker_raw_image, cv2.COLOR_RGB2BGR)

        # Create a ColorChecker detector
        detector = cv2.mcc.CCheckerDetector_create()
    
        # Process the image to detect the ColorChecker
        detected = detector.process(imageBGR, cv2.mcc.MCC24, 1)
        
        if not detected:
            return None

        # Get the list of detected ColorCheckers
        checkers = detector.getListColorChecker()
        
        for checker in checkers:
            # Create a CCheckerDraw object to visualize the ColorChecker
            if drawPatches:
                cdraw = cv2.mcc.CCheckerDraw_create(checker)
                cdraw.draw(imageBGR)
                self.img_draw = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

            # Get the detected color patches and rearrange them
            chartsRGB = checker.getChartsRGB()
            width, height = chartsRGB.shape[:2]
            self.color_patches = chartsRGB[:, 1].copy().reshape(int(width / 3), 1, 3) / 255.0
    
    def setPathcalibrationParams(self, file_path) -> None:
        self.params_path = file_path

    def saveCalibrationParams(self) -> None:
        # Save the color patches and configuration to a pickle file
        params = {
            'color_patches': self.color_patches
        }
        with open(self.params_path, 'wb') as f:
            pickle.dump(params, f)
    
    def reconstructModelFromParams(self):
        # Load the color patches and configuration from a pickle file
        try:
            with open(self.params_path, 'rb') as f:
                params = pickle.load(f)
        except FileNotFoundError:
            self.params_path = "resources/calibration/default_calibration_params.pickle"
            self.config.setValue("calibration",
                                "calibration_params_path",
                                "resources/calibration/default_calibration_params.pickle")
            with open(self.params_path, 'rb') as f:
                params = pickle.load(f)
        except Exception as e:
            print("Error")

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

    def clearAllCalibration(self) -> None:
        self.color_checker_raw_image = None
        self.color_patches = None
        self.img_draw = None
    
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
        self.calibrated_image = None

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
    def loadImage(self, image, calibration) -> None:
        self.original_image = self.cropSquare(image)
        model = calibration.reconstructModelFromParams()
        self.calibrated_image = calibration.applyColorCorrection(self.original_image, model)
        self.original_image_size = self.calibrated_image.shape[:2]
        self.scaled_image = self.decimateImage(self.calibrated_image)
        self.predictor.set_image(self.scaled_image)

    def saveOriginalImage(self, path) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".png":
            cv2.imwrite(path, cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR))
        elif ext == ".npy":
            np.save(path, self.original_image)
        else:
            QMessageBox.warning(self, "Formato no soportado",
                                f"El formato {ext} no es válido para guardar.")
    
    def saveCalibratedImage(self, path) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".png":
            cv2.imwrite(path, cv2.cvtColor(self.calibrated_image, cv2.COLOR_RGB2BGR))
        elif ext == ".npy":
            np.save(path, self.calibrated_image)
        else:
            QMessageBox.warning(self, "Formato no soportado",
                                f"El formato {ext} no es válido para guardar.")
    
    def getScaledImage(self) -> np.ndarray:
        return self.scaled_image
        
    def getRawMask(self):
        return self.raw_mask
    
    def getFeatheredMask(self):
        return self.feathered_mask
    
    def getScaledFeatheredMask(self):
        return self.scaled_feathered_mask
    
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
        guide_I = self.calibrated_image.astype(np.float32)/255.0
        mask_p = self.raw_mask.astype(np.float32)

        self.feathered_mask = cv2.ximgproc.guidedFilter(
                            guide=guide_I,
                            src=mask_p,
                            radius=radius,
                            eps=eps
                        )
        self.scaled_feathered_mask = self.decimateImage(self.feathered_mask)

    def getSegmentedRegionHistogram(self):
        if self.feathered_mask is None or self.calibrated_image is None:
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
            hist = cv2.calcHist([self.calibrated_image], [i], binary_mask_uint8, [256], [0, 256])
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
        # Convertir a LAB en rangos estándar
        image_lab = self.fromRGBtoLAB(self.calibrated_image)

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


    def estimateColorByKMeans(self, n_clusters: int = 3,
                                min_weight_threshold: float = 0.3) -> np.ndarray:
        # Convertir a LAB en rangos estándar
        image_lab = self.fromRGBtoLAB(self.calibrated_image)
        
        # Filtrar pixels con peso significativo
        valid_mask: bool = self.feathered_mask > min_weight_threshold
        pixels_lab = image_lab[valid_mask]
        weights = self.feathered_mask[valid_mask]

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
        cluster_weights = np.zeros(n_clusters, dtype=np.float32)
        for i in range(n_clusters):
            cluster_mask: bool = (labels == i)
            cluster_weights[i] = np.sum(weights[cluster_mask])
        
        # Seleccionar cluster con mayor peso acumulado
        dominant_cluster_idx = np.argmax(cluster_weights)
        dominant_color_lab = centroids[dominant_cluster_idx]
        
        self.findNearestPantone(dominant_color_lab)

    
    def estimateColorBySoftVoting(self,
                                sigma: float = 10.0,
                                n_clusters: int = 100,
                                min_weight_threshold: float = 0.1) -> np.ndarray:
        # Convertir a LAB en rangos estándar
        image_lab = self.fromRGBtoLAB(self.calibrated_image)

        # Filtrar pixels con peso significativo
        valid_mask: bool = self.feathered_mask > min_weight_threshold
        pixels_lab = image_lab[valid_mask]
        weights = self.feathered_mask[valid_mask]

        # Criterios de parada para K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
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
        cluster_weights = np.zeros(n_clusters, dtype=np.float32)
        for i in range(n_clusters):
            cluster_mask: bool = (labels == i)
            cluster_weights[i] = np.sum(weights[cluster_mask])

        # Normalizar pesos
        cluster_weights = cluster_weights / np.sum(cluster_weights)

        # Soft Voting Probabilístico (Kernel RBF)
        n_pantone = len(self.pantone_lab_colors)

        distances_ciede = np.zeros((n_clusters, n_pantone), dtype=np.float32)

        for i in range(n_clusters):
            centroid_lab = centroids[i].reshape(1, 3)
            distances_ciede[i] = deltaE_ciede2000(centroid_lab, self.pantone_lab_colors)
        
        probabilities = np.exp(-(distances_ciede**2) / (2 * sigma**2))
        pantone_scores = cluster_weights.dot(probabilities)

        top_3_indx = np.argsort(pantone_scores)[-3:][::-1]
        for i, indx in enumerate(top_3_indx):
            name = self.pantone_name_colors[indx]
            lab = self.pantone_lab_colors[indx]
            rgb = np.astype(np.round(lab2rgb(lab)*255), np.uint8)
            self.top_colors[f"Color {i+1}"] = {
                                            "Pantone Name": name,
                                            "LAB": lab,
                                            "RGB": rgb,
                                            "ΔE00": -1
                                            }

    @staticmethod
    def computeDeltaE1976(pixels_lab: np.ndarray, 
                            pantones_lab: np.ndarray) -> np.ndarray:
        """
        Calcula Delta E 1976 (distancia euclidiana en LAB) vectorizada.
        Extremadamente rápido (~1000x más rápido que CIEDE2000).
        
        Args:
            pixels_lab: (n_pixels, 3)
            pantones_lab: (n_pantones, 3)
        
        Returns:
            distances: (n_pixels, n_pantones) - distancias euclidianas
        """
        # Broadcasting: (n_pixels, 1, 3) - (1, n_pantones, 3)
        diff = pixels_lab[:, np.newaxis, :] - pantones_lab[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        return distances
    
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
        self.scene = None
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
    
    def loadScene(self):
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

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

    def getImageArrayFromScene(self):
        # Obtener el rectángulo de la escena
        scene_rect = self.scene.sceneRect()
        
        # Crear una imagen con el tamaño de la escena
        width = int(scene_rect.width())
        height = int(scene_rect.height())
        
        # Crear QImage con formato RGB32
        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(0xFFFFFF)  # Fondo blanco (opcional)
        
        # Crear painter y renderizar la escena
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        self.scene.render(painter, QRectF(image.rect()), scene_rect)
        painter.end()
        
        # Convertir QImage a numpy array
        # Obtener puntero a los datos de la imagen
        ptr = image.constBits()
        arr = np.array(ptr).reshape(height, width, 4)  # RGBA
        
        # Convertir de BGRA a RGB (Qt usa BGRA internamente)
        rgb_array = arr[:, :, [2, 1, 0]].copy()  # Intercambiar canales B y R
        
        return rgb_array

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
        self.style_path = "resources/styles/dark_theme.qss"
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
    
    def getConfig(self):
        """Retorna la configuración actual"""
        return self.config
    
    def getValue(self, category, key):
        """Obtiene un valor específico de la configuración"""
        return self.config.get(category, {}).get(key)
    
    def setValue(self, category: str, key: str, value) -> None:
        """
        Cambia el valor de una llave específica en la configuración y guarda el cambio.
        Si la categoría o la llave no existen, las crea.
        """
        if self.config is None:
            self.loadConfig()
        
        # Crear categoría si no existe
        if category not in self.config:
            self.config[category] = {}
        
        # Asignar el nuevo valor
        self.config[category][key] = value

        # Guardar automáticamente en el archivo de usuario
        self.saveUserConfig(self.config)
    
    def resetToDefault(self):
        """Resetea a la configuración por defecto y la guarda"""
        self.loadDefaultConfig()
        # Eliminar configuración de usuario
        if os.path.exists(self.user_config_path):
            os.remove(self.user_config_path)
    
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
    
    def __init__(self, parent=None, config_manager=None, processing=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.processing = processing

        self.color_mask = self.processing.getFeatheredMaskColor().tolist()
        
        # Configuración de la ventana
        self.setWindowTitle("Configuración")
        self.setMinimumSize(550, 440)

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
        self.tab_widget.addTab(self.createCameraTab(), "Cámara")
        self.tab_widget.addTab(self.createCalibrationTab(), "Calibración")
        self.tab_widget.addTab(self.createSegmentationTab(), "Segmentación")
        self.tab_widget.addTab(self.createColorTab(), "Color")
        self.tab_widget.addTab(self.createExportTab(), "Exportar")
        
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
        camera_layout.setHorizontalSpacing(20)
        
        # Auto gain (CheckBox)
        self.check_auto_gain = QCheckBox()
        self.check_auto_gain.setChecked(True)
        camera_layout.addRow("Ganancia automática", self.check_auto_gain)
        
        # Preset gain (ComboBox)
        self.combo_preset_gain = QComboBox()
        self.combo_preset_gain.addItems(["Low", "Medium", "High"])
        camera_layout.addRow("Preset gain", self.combo_preset_gain)
        
        # Exposition time (SpinBox en microsegundos)
        self.spin_exposition_time = QSpinBox()
        self.spin_exposition_time.setRange(100, 1000000)  # 100µs a 1s
        self.spin_exposition_time.setValue(10000)
        self.spin_exposition_time.setSuffix(" µs")
        self.spin_exposition_time.setSingleStep(1000)
        camera_layout.addRow("Tiempo de exposición", self.spin_exposition_time)
        
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
        
        # Calibration params file path
        params_layout = QHBoxLayout()
        btn_calib_params = QPushButton("Seleccionar Parámetros")
        btn_calib_params.clicked.connect(self.selectCalibParamsPath)
        self.label_calib_params_path = QLabel("No seleccionado")
        self.label_calib_params_path.setWordWrap(True)
        self.label_calib_params_path.setStyleSheet("font-size: 11px; color: #a0a0a0;")
        params_layout.addWidget(btn_calib_params)
        params_layout.addWidget(self.label_calib_params_path, 1)
        
        calibration_layout.addWidget(QLabel("Archivo parámetros de calibración"))
        calibration_layout.addLayout(params_layout)
        
        calibration_group.setLayout(calibration_layout)
        layout.addWidget(calibration_group)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
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
        scene_group = QGroupBox("Escena")
        scene_layout = QFormLayout()
        scene_layout.setHorizontalSpacing(20)
        
        # Image scale size
        self.spin_image_scale = QSpinBox()
        self.spin_image_scale.setRange(256, 1920)
        self.spin_image_scale.setValue(720)
        self.spin_image_scale.setSingleStep(64)
        self.spin_image_scale.setSuffix(" px")
        scene_layout.addRow("Tamaño imagen escalada", self.spin_image_scale)
        
        # Marker radius
        self.spin_marker_radius = QSpinBox()
        self.spin_marker_radius.setRange(1, 20)
        self.spin_marker_radius.setValue(1)
        self.spin_marker_radius.setSuffix(" px")
        scene_layout.addRow("Radio punto", self.spin_marker_radius)
        
        scene_group.setLayout(scene_layout)
        layout.addWidget(scene_group)
        
        # Group Box "Mask"
        mask_group = QGroupBox("Máscara")
        mask_layout = QFormLayout()
        mask_layout.setHorizontalSpacing(20)
        
        # Mask color (Frame + Button)
        mask_color_layout = QHBoxLayout()
        mask_color_layout.setSpacing(10)
        self.frame_mask_color = QFrame()
        self.frame_mask_color.setFixedSize(60, 30)
        self.frame_mask_color.setFrameStyle(QFrame.Box)
        self.frame_mask_color.setStyleSheet("background-color: rgb(158, 16, 127); border: 2px solid #999;")
        
        btn_mask_color = QPushButton("Cambiar Color")
        btn_mask_color.clicked.connect(self.selectMaskColor)
        
        mask_color_layout.addWidget(self.frame_mask_color)
        mask_color_layout.addWidget(btn_mask_color)
        mask_color_layout.addStretch()
        
        mask_layout.addRow("Color máscara", mask_color_layout)
        
        mask_group.setLayout(mask_layout)
        layout.addWidget(mask_group)
        
        # Group Box "Guided Filter"
        filter_group = QGroupBox("Filtro Guiado")
        filter_layout = QFormLayout()
        filter_layout.setHorizontalSpacing(20)
        
        # Radius
        self.spin_filter_radius = QSpinBox()
        self.spin_filter_radius.setRange(1, 50)
        self.spin_filter_radius.setValue(2)
        self.spin_filter_radius.setSuffix(" px")
        filter_layout.addRow("Radio ventana", self.spin_filter_radius)
        
        # Epsilon
        self.spin_filter_epsilon = QDoubleSpinBox()
        self.spin_filter_epsilon.setRange(0.01, 1.0)
        self.spin_filter_epsilon.setValue(0.3)
        self.spin_filter_epsilon.setDecimals(2)
        self.spin_filter_epsilon.setSingleStep(0.01)
        self.spin_filter_epsilon.setSuffix("²")
        filter_layout.addRow("Epsilon (ε²)", self.spin_filter_epsilon)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def selectMaskColor(self):
        """Abre diálogo para seleccionar color de máscara"""
        r,g,b = self.processing.getFeatheredMaskColor().tolist()
        self.color_mask = [r, g, b]
        current_color = QColor(r,g,b)
        # Abrir diálogo de color
        new_color = QColorDialog.getColor(current_color, self, "Seleccionar color de máscara")
        
        if new_color.isValid():
            r, g, b = new_color.red(), new_color.green(), new_color.blue()
            self.color_mask = [r, g, b]
            #self.processing.setFeatheredMaskColor([r,g,b])
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
        btn_pantone_file = QPushButton("Seleccionar Base de Datos")
        btn_pantone_file.clicked.connect(self.selectPantoneDatabase)
        self.label_pantone_path = QLabel("No seleccionado")
        self.label_pantone_path.setWordWrap(True)
        self.label_pantone_path.setStyleSheet("font-size: 11px; color: #a0a0a0;")
        pantone_file_layout.addWidget(btn_pantone_file)
        pantone_file_layout.addWidget(self.label_pantone_path, 1)
        
        pantone_layout.addWidget(QLabel("Archivo base de datos Pantone"))
        pantone_layout.addLayout(pantone_file_layout)
        
        pantone_group.setLayout(pantone_layout)
        layout.addWidget(pantone_group)
        
        # Group Box "Color Method"
        method_group = QGroupBox("Método de color")
        method_layout = QVBoxLayout()
        
        # Select method (ComboBox)
        method_select_layout = QFormLayout()
        method_select_layout.setHorizontalSpacing(20)
        self.combo_color_method = QComboBox()
        self.combo_color_method.addItems(["Mediana", "K-Means", "Soft Voting"])
        self.combo_color_method.currentTextChanged.connect(self.onColorMethodChanged)
        method_select_layout.addRow("Seleccionar método", self.combo_color_method)
        method_layout.addLayout(method_select_layout)
        
        # Contenedor para parámetros condicionales
        self.method_params_layout = QFormLayout()
        self.method_params_layout.setHorizontalSpacing(20)
        
        # Parámetros para Median
        self.label_threshold_median = QLabel("Umbral máscara")
        self.spin_threshold_median = QDoubleSpinBox()
        self.spin_threshold_median.setRange(0.0, 1.0)
        self.spin_threshold_median.setValue(0.0)
        self.spin_threshold_median.setDecimals(2)
        self.spin_threshold_median.setSingleStep(0.05)
        self.method_params_layout.addRow(self.label_threshold_median, self.spin_threshold_median)
        
        # Parámetros para KMeans
        self.label_threshold_kmeans = QLabel("Umbral máscara")
        self.spin_threshold_kmeans = QDoubleSpinBox()
        self.spin_threshold_kmeans.setRange(0.0, 1.0)
        self.spin_threshold_kmeans.setValue(0.3)
        self.spin_threshold_kmeans.setDecimals(2)
        self.spin_threshold_kmeans.setSingleStep(0.05)
        self.method_params_layout.addRow(self.label_threshold_kmeans, self.spin_threshold_kmeans)
        
        self.label_number_clusters = QLabel("Número clusters")
        self.spin_number_clusters = QSpinBox()
        self.spin_number_clusters.setRange(2, 10)
        self.spin_number_clusters.setValue(3)
        self.method_params_layout.addRow(self.label_number_clusters, self.spin_number_clusters)
        
        # Parámetros para SoftVoting
        self.label_threshold_softvoting = QLabel("Umbral máscara")
        self.spin_threshold_softvoting = QDoubleSpinBox()
        self.spin_threshold_softvoting.setRange(0.0, 1.0)
        self.spin_threshold_softvoting.setValue(0.1)
        self.spin_threshold_softvoting.setDecimals(2)
        self.spin_threshold_softvoting.setSingleStep(0.05)
        self.method_params_layout.addRow(self.label_threshold_softvoting, self.spin_threshold_softvoting)

        self.label_n_clusters_soft = QLabel("Número clusters")
        self.spin_n_clusters_soft = QSpinBox()
        self.spin_n_clusters_soft.setRange(3, 1000)
        self.spin_n_clusters_soft.setValue(100)
        self.spin_n_clusters_soft.setSingleStep(1)
        self.method_params_layout.addRow(self.label_n_clusters_soft, self.spin_n_clusters_soft)
        
        self.label_sigma = QLabel("Sigma")
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(1.0, 100.0)
        self.spin_sigma.setValue(10.0)
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setSingleStep(1.0)
        self.method_params_layout.addRow(self.label_sigma, self.spin_sigma)
        
        method_layout.addLayout(self.method_params_layout)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        
        # Mostrar solo los parámetros del método por defecto
        self.onColorMethodChanged("Mediana")
        
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
        self.label_n_clusters_soft.hide()
        self.spin_n_clusters_soft.hide()
        
        # Mostrar solo los parámetros del método seleccionado
        if method == "Mediana":
            self.label_threshold_median.show()
            self.spin_threshold_median.show()
        elif method == "K-Means":
            self.label_threshold_kmeans.show()
            self.spin_threshold_kmeans.show()
            self.label_number_clusters.show()
            self.spin_number_clusters.show()
        elif method == "Soft Voting":
            self.label_threshold_softvoting.show()
            self.spin_threshold_softvoting.show()
            self.label_sigma.show()
            self.spin_sigma.show()
            self.label_n_clusters_soft.show()
            self.spin_n_clusters_soft.show()
    
    # ==================== TAB 5: EXPORT ====================
    def createExportTab(self):
        """Pestaña para configurar parámetros de exportación"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Group Box "Imágenes Análisis"
        analysis_group = QGroupBox("Imágenes")
        analysis_layout = QFormLayout()
        analysis_layout.setHorizontalSpacing(20)
        
        # Resolution DPI
        self.spin_export_dpi = QSpinBox()
        self.spin_export_dpi.setRange(72, 600)
        self.spin_export_dpi.setValue(300)
        self.spin_export_dpi.setSingleStep(50)
        self.spin_export_dpi.setSuffix(" dpi")
        analysis_layout.addRow("Resolución DPI", self.spin_export_dpi)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Group Box "Resultados"
        results_group = QGroupBox("Resultados")
        results_layout = QVBoxLayout()
        results_layout.setSpacing(10)

        # Save original image (CheckBox)
        check_save_widget = QWidget()
        check_save_layout = QFormLayout(check_save_widget)
        check_save_layout.setHorizontalSpacing(20)
        check_save_layout.setContentsMargins(0, 0, 0, 0) 
        self.check_save_original_img = QCheckBox()
        self.check_save_original_img.setChecked(True)
        check_save_layout.addRow("Guardar imagenes originales", self.check_save_original_img)
        results_layout.addWidget(check_save_widget)

        # Folder Resultados
        folder_layout = QHBoxLayout()
        btn_results_folder = QPushButton("Seleccionar Carpeta")
        btn_results_folder.clicked.connect(self.selectResultsFolder)
        self.label_results_folder = QLabel("No seleccionado")
        self.label_results_folder.setWordWrap(True)
        self.label_results_folder.setStyleSheet("font-size: 11px; color: #a0a0a0;")
        folder_layout.addWidget(btn_results_folder)
        folder_layout.addWidget(self.label_results_folder, 1)
        
        results_layout.addWidget(QLabel("Carpeta de resultados"))
        results_layout.addLayout(folder_layout)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def selectResultsFolder(self):
        """Abre diálogo para seleccionar carpeta de resultados"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar Carpeta de Resultados",
            ".",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder:
            self.label_results_folder.setText(folder)
    
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
        self.spin_n_clusters_soft.setValue(config['color']['method']['n_clusters_soft'])
        
        # Export
        self.spin_export_dpi.setValue(config['export']['analysis_images']['dpi'])
        self.check_save_original_img.setChecked(config['export']['results']['save_original_img'])
        self.label_results_folder.setText(config['export']['results']['folder_path'])
    
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

        mask_color = self.color_mask

        dict_values: dict = {
                                'camera': {
                                    'auto_gain': self.check_auto_gain.isChecked(),
                                    'preset_gain': self.combo_preset_gain.currentText(),
                                    'exposition_time': self.spin_exposition_time.value(),
                                    'description': 'Parámetros de la cámara'
                                },
                                'calibration': {
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
                                        'n_clusters_soft': self.spin_n_clusters_soft.value()
                                    },
                                    'description': 'Parámetros de estimación de color'
                                },
                                'export': {
                                    'analysis_images': {
                                        'dpi': self.spin_export_dpi.value()
                                    },
                                    'results': {
                                        'folder_path': self.label_results_folder.text(),
                                        'save_original_img': self.check_save_original_img.isChecked()
                                    },
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

        # Inicializar el gestor de configuración
        self.config_manager = ConfigManager()

        self.calibration = Calibration(self.config_manager)
        self.processing = ImageProcessing()

        # Window Settings
        self.setWindowTitle("SACISMC")

        # Widgets
        self.viewer = Viewer()
        self.viewer.setFixedSize(720, 720)

        # Panel lateral derecho
        self.side_panel = QStackedWidget()
        self.side_panel.setFixedWidth(440)
        self.operationWidget = self.createSideOperationWidget()
        self.side_panel.addWidget(self.operationWidget)

        # Menu Bar
        menu_bar = self.menuBar()
        menu_file = menu_bar.addMenu("Archivo")
        action_open_img = menu_file.addAction("Abrir imagen")
        action_open_img.triggered.connect(self.openImage)

        submenu_extract = menu_file.addMenu("Exportar")
        action_extract_img = submenu_extract.addAction("Imagen")
        action_extract_img.triggered.connect(self.saveImage)
        action_extract_filter_comparison = submenu_extract.addAction("Comparación filtro guiado")
        action_extract_filter_comparison.triggered.connect(self.exportFeatheredComparisonImage)
        action_extract_histogram = submenu_extract.addAction("Histograma de color")
        action_extract_histogram.triggered.connect(self.exportHistogram)

        action_settings = menu_file.addAction("Configuración")
        action_settings.triggered.connect(self.openSettingsDialog)

        menu_delete = menu_bar.addMenu("Eliminar")
        action_delete_last_point = menu_delete.addAction("Último punto")
        action_delete_last_point.triggered.connect(self.viewer.clearLastPoint)
        action_delete_all_points = menu_delete.addAction("Todos los puntos")
        action_delete_all_points.triggered.connect(self.viewer.clearAllPoints)
        action_delete_mask = menu_delete.addAction("Máscara")
        action_delete_mask.triggered.connect(self.viewer.clearMask)
        
        menu_calibrate = menu_bar.addMenu("Calibrar")
        action_select_checker = menu_calibrate.addAction("Iniciar calibración de color")
        action_select_checker.triggered.connect(self.startColorCalibration)

        # Main Layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10) 
        main_layout.setSpacing(20)
        main_layout.addWidget(self.viewer)
        main_layout.addWidget(self.side_panel)
        main_layout.addStretch()

        # Containers and Layouts
        main_container = QWidget()
        main_container.setObjectName("MainContainer")
        main_container.setLayout(main_layout)
        self.setCentralWidget(main_container)

        # Styles
        self.setStyleSheet(self.config_manager.getStyle())

        # Initial config
        self.applyConfigToComponents(self.config_manager.getConfig())

    # Methods

    def openSettingsDialog(self) -> None:
        """Abre el diálogo de configuración"""
        dialog = ConfigDialog(self, self.config_manager, self.processing)
        
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
        self.calibration.params_path = config['calibration']['calibration_params_path']
        
        # Aplicar a Viewer
        self.viewer.min_markpoint_radius = config['segmentation']['scene']['marker_radius']

        # Crear carpeta de resultados si no existe
        results_folder = config['export']['results']['folder_path']
        os.makedirs(results_folder, exist_ok=True)
        
        # Crear subcarpeta images
        images_folder = os.path.join(results_folder, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # Aplicar a Processing si existe
        if self.processing:
            self.processing.scaled_image_size = config['segmentation']['scene']['image_scale_size']
            self.processing.r_filter = config['segmentation']['guided_filter']['radius']
            self.processing.eps_filter = config['segmentation']['guided_filter']['epsilon']**2
            self.processing.setFeatheredMaskColor(config['segmentation']['mask']['mask_color'])
            self.processing.pantone_database_path = config['color']['pantone']['database_file']
            self.processing.loadPantoneDatabase()
            # Cambiar color mascara
            if self.viewer.mask_item:
                self.viewer.clearMask()
                f_color = self.processing.getFeatheredMaskColor()
                feathered_mask_colored = self.processing.createColoredMask(self.processing.getScaledFeatheredMask(), f_color)
                self.viewer.addOverlay(self.viewer.fromCV2ToQPixmap(feathered_mask_colored))

    def createSideOperationWidget(self) -> QVBoxLayout:
        """Crea el layout lateral con controles"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        
        # Botones
        self.group_box_button = QWidget()
        group_button_layout = QVBoxLayout()
        group_button_layout.setContentsMargins(0, 0, 0, 0) 
        group_button_layout.setSpacing(6)

        self.button_take_photo = QPushButton("Tomar Foto")
        self.button_take_photo.clicked.connect(self.takePhotoOperationClicked)
        self.button_take_photo.setFixedHeight(50) #40
        group_button_layout.addWidget(self.button_take_photo)

        self.button_run = QPushButton("Segmentar")
        self.button_run.clicked.connect(self.runSegmentation)
        self.button_run.setFixedHeight(50) #40
        group_button_layout.addWidget(self.button_run)

        self.group_box_button.setLayout(group_button_layout)
        layout.addWidget(self.group_box_button)

        self.group_box_others = QWidget()
        group_others_layout = QVBoxLayout()
        group_others_layout.setContentsMargins(0, 0, 0, 0) 
        group_others_layout.setSpacing(10)

        # -------------------- DISPLAY --------------------

        group_box_view = QGroupBox("Visualización")
        group_layout_view = QVBoxLayout()
        group_layout_view.setContentsMargins(13, 15, 13, 10) 
        group_layout_view.setSpacing(7)
        
        # Checkbox
        self.checkbox_show_mask = QCheckBox("Mostrar máscara")
        self.checkbox_show_mask.setChecked(True)
        self.checkbox_show_mask.stateChanged.connect(self.showMask)
        group_layout_view.addWidget(self.checkbox_show_mask)

        self.checkbox_show_points = QCheckBox("Mostrar puntos")
        self.checkbox_show_points.setChecked(True)
        self.checkbox_show_points.stateChanged.connect(self.showPoints)
        group_layout_view.addWidget(self.checkbox_show_points)

        group_box_view.setLayout(group_layout_view)
        group_others_layout.addWidget(group_box_view)

        # -------------------- COLOR --------------------

        group_box_colors = QGroupBox("Color")
        group_layout_colors = QVBoxLayout()
        group_layout_colors.setContentsMargins(13, 15, 13, 10) 
        group_layout_colors.setSpacing(10)

        color_1_group_box_style = """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #707070;
                border-radius: 7px;
                margin-top: 1ex;
                margin-right: 0px;
                margin-bottom: 0px;
                margin-left: 0px;
                padding: 13px 0px 0px 0px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 0px;
                padding: 0px 3px 0px 0px;
                margin: 0px 3px 0px 0px;
                color: #e0e0e0;
            }
        """
        color_others_group_box_style = """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #707070;
                border-radius: 7px;
                margin-top: 1ex;
                margin-right: 0px;
                margin-bottom: 0px;
                margin-left: 0px;
                padding: 15px 0px 0px 0px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 0px;
                padding: 0px 3px 0px 0px;
                margin: 0px 3px 0px 0px;
                color: #707070;
            }
        """
        color_display_buttons_style = """
            QPushButton {
                font-weight: bold;
                font-size: 24px;
                padding: 5px;
                margin: 0px;
                border: 1px solid #707070;
                border-radius: 7px;
                background-color: #424242;
                color: #e0e0e0;
            }
        """

        # ---- COLOR 1 ----
        self.group_box_color_1 = QGroupBox("")
        group_layout_color_1 = QHBoxLayout(self.group_box_color_1)
        group_layout_color_1.setContentsMargins(10, 5, 10, 10) 
        self.group_box_color_1.setStyleSheet(color_1_group_box_style)

        # Left
        left_widget_color_1 = QWidget()
        left_layout_color_1 = QVBoxLayout(left_widget_color_1)
        left_layout_color_1.setContentsMargins(0, 2, 0, 2)

        self.color_lab_1 = QLabel("LAB")
        self.color_lab_1.setStyleSheet("font-weight: normal;")
        left_layout_color_1.addWidget(self.color_lab_1)

        self.color_delta_1 = QLabel("ΔE00")
        self.color_delta_1.setStyleSheet("font-weight: normal;")
        left_layout_color_1.addWidget(self.color_delta_1)
        
        # Right
        right_widget_color_1 = QWidget()
        right_layout_color_1 = QVBoxLayout(right_widget_color_1)
        right_layout_color_1.setContentsMargins(0, 0, 0, 0) 
        
        self.color_display_1 = QPushButton("")
        self.color_display_1.setFixedHeight(50)
        self.color_display_1.setStyleSheet(color_display_buttons_style)


        right_layout_color_1.addWidget(self.color_display_1)

        group_layout_color_1.addWidget(left_widget_color_1)
        group_layout_color_1.addWidget(right_widget_color_1)

        # ---- OTHER COLORS ----
        other_colors_widget = QWidget()
        other_colors_layout = QHBoxLayout(other_colors_widget)
        other_colors_layout.setContentsMargins(0, 0, 0, 0) 
        other_colors_layout.setSpacing(10)

        # Color 2
        self.group_box_color_2 = QGroupBox("")
        group_layout_color_2 = QVBoxLayout(self.group_box_color_2)
        group_layout_color_2.setContentsMargins(10, 5, 10, 10) 
        self.group_box_color_2.setStyleSheet(color_others_group_box_style)

        self.color_lab_2 = QLabel("LAB")
        self.color_lab_2.setStyleSheet("font-weight: normal;")
        group_layout_color_2.addWidget(self.color_lab_2)

        self.color_delta_2 = QLabel("ΔE00")
        self.color_delta_2.setStyleSheet("font-weight: normal;")
        group_layout_color_2.addWidget(self.color_delta_2)
        
        self.color_display_2 = QPushButton("")
        self.color_display_2.setFixedHeight(40)
        self.color_display_2.setStyleSheet(color_display_buttons_style)

        group_layout_color_2.addWidget(self.color_display_2)

        # Color 3
        self.group_box_color_3 = QGroupBox("")
        group_layout_color_3 = QVBoxLayout(self.group_box_color_3)
        group_layout_color_3.setContentsMargins(10, 5, 10, 10) 
        self.group_box_color_3.setStyleSheet(color_others_group_box_style)

        self.color_lab_3 = QLabel("LAB")
        self.color_lab_3.setStyleSheet("font-weight: normal;")
        group_layout_color_3.addWidget(self.color_lab_3)

        self.color_delta_3 = QLabel("ΔE00")
        self.color_delta_3.setStyleSheet("font-weight: normal;")
        group_layout_color_3.addWidget(self.color_delta_3)
        
        self.color_display_3 = QPushButton("")
        self.color_display_3.setFixedHeight(40)
        self.color_display_3.setStyleSheet(color_display_buttons_style)

        group_layout_color_3.addWidget(self.color_display_3)


        other_colors_layout.addWidget(self.group_box_color_2)
        other_colors_layout.addWidget(self.group_box_color_3)

        group_layout_colors.addWidget(self.group_box_color_1)
        group_layout_colors.addWidget(other_colors_widget)
        group_box_colors.setLayout(group_layout_colors)
        group_others_layout.addWidget(group_box_colors)


        line = QFrame()
        line.setFrameShape(QFrame.NoFrame)
        line.setFixedHeight(1)
        group_others_layout.addWidget(line)

        # -------------------- REGISTER --------------------

        group_box_3 = QGroupBox("Registro")
        group_layout_3 = QVBoxLayout()
        group_layout_3.setContentsMargins(13, 15, 13, 10) 
        group_layout_3.setSpacing(10)

        self.input_id = QLineEdit()
        self.input_id.setPlaceholderText("Identificador")
        self.input_id.setFixedHeight(40)
        group_layout_3.addWidget(self.input_id)

        self.log_button = QPushButton("Guardar")
        self.log_button.clicked.connect(self.saveRecord)
        self.log_button.setFixedHeight(50)
        self.log_button.setStyleSheet("font-size: 16px;")
        group_layout_3.addWidget(self.log_button)

        group_box_3.setLayout(group_layout_3)
        group_others_layout.addWidget(group_box_3)


        self.group_box_others.setLayout(group_others_layout)
        self.group_box_others.hide()

        layout.addWidget(self.group_box_others)
        layout.addStretch()

        return widget
    
    def createSideCalibrationWidget(self) -> QVBoxLayout:
        # Layout total
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ------------ Group Box - Load Image ------------
        group_box_load_image = QGroupBox("Tabla Macbeth ColorChecker")
        layout_load_image = QVBoxLayout(group_box_load_image)
        layout_load_image.setContentsMargins(13, 15, 13, 10)
        layout_load_image.setSpacing(7)

        label_load_image = QLabel("Elige una opción para cargar la tabla:")
        layout_load_image.addWidget(label_load_image)

        line1 = QFrame()
        line1.setFrameShape(QFrame.NoFrame)
        line1.setFixedHeight(1)
        layout_load_image.addWidget(line1)

        # ------ Group Box - Take Photo
        group_box_take_photo = QGroupBox("Tomar Foto")
        layout_take_photo = QVBoxLayout(group_box_take_photo)
        layout_take_photo.setContentsMargins(13, 15, 13, 10)
        layout_take_photo.setSpacing(7)

        label_take_photo_step_1 = QLabel("1. Ubica la tabla al interior del domo")
        layout_take_photo.addWidget(label_take_photo_step_1)
        label_take_photo_step_2 = QLabel("2. Asegura la correcta orientación de la tabla")
        layout_take_photo.addWidget(label_take_photo_step_2)

        button_take_photo = QPushButton("Capturar")
        button_take_photo.clicked.connect(self.takePhotoCalibrationClicked)
        button_take_photo.setFixedHeight(40)
        layout_take_photo.addWidget(button_take_photo)

        layout_load_image.addWidget(group_box_take_photo)

        # ------ o
        label_or = QLabel("-   o   -")
        label_or.setStyleSheet("color: #e0e0e0; font-size: 13px; font-weight: normal;")
        label_or.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout_load_image.addWidget(label_or)

        # ------ Group Box - Select Photo
        group_box_select_photo = QGroupBox("Seleccionar Foto")
        layout_select_photo = QVBoxLayout(group_box_select_photo)
        layout_select_photo.setContentsMargins(13, 15, 13, 10)
        layout_select_photo.setSpacing(7)

        label_select_photo = QLabel("Desde el explorador de archivos, carga la foto de la tabla")
        layout_select_photo.addWidget(label_select_photo)

        button_select_photo = QPushButton("Cargar")
        button_select_photo.clicked.connect(self.selectPhotoCalibrationClicked)
        button_select_photo.setFixedHeight(40)
        layout_select_photo.addWidget(button_select_photo)

        layout_load_image.addWidget(group_box_select_photo)

        layout.addWidget(group_box_load_image)

        # ------------ Group Box - Calibration Params ------------
        self.group_box_params = QGroupBox("Parámetros de Calibración")
        layout_params = QVBoxLayout(self.group_box_params)
        layout_params.setContentsMargins(13, 10, 13, 10)
        layout_params.setSpacing(10)

        # ------ Group Box - Detect
        group_box_detect = QGroupBox("")
        group_box_detect.setStyleSheet("padding-top: 0px;")
        layout_detect = QVBoxLayout(group_box_detect)
        layout_detect.setContentsMargins(13, 5, 13, 10)
        layout_detect.setSpacing(7)

        label_detect = QLabel("Detectar parches de color de la tabla")
        layout_detect.addWidget(label_detect)

        button_detect = QPushButton("Detectar")
        button_detect.clicked.connect(self.detectColorCheckerClicked)
        button_detect.setFixedHeight(40)
        layout_detect.addWidget(button_detect)

        layout_params.addWidget(group_box_detect)

        # ------ Group Box - Save and Apply
        self.group_box_save_apply = QGroupBox("")
        self.group_box_save_apply.setStyleSheet("padding-top: 0px;")
        layout_save_apply = QVBoxLayout(self.group_box_save_apply)
        layout_save_apply.setContentsMargins(13, 5, 13, 10)
        layout_save_apply.setSpacing(7)

        #input_file_name = QLineEdit()
        #input_file_name.setPlaceholderText("Nombre archivo .pickle")
        #input_file_name.setFixedHeight(30)
        #layout_save_apply.addWidget(input_file_name)

        label_save_apply = QLabel("Guardar archivo y aplicar cambios")
        layout_save_apply.addWidget(label_save_apply)

        button_save_apply = QPushButton("Guardar y Aplicar")
        button_save_apply.clicked.connect(self.saveAndApplyClicked)
        button_save_apply.setFixedHeight(40)
        layout_save_apply.addWidget(button_save_apply)

        self.group_box_save_apply.hide()
        layout_params.addWidget(self.group_box_save_apply)

        self.group_box_params.hide()
        layout.addWidget(self.group_box_params)

        # ------------ Radio Buttons ------------
        self.radio_buttons = QWidget()
        layout_radio_buttons = QHBoxLayout(self.radio_buttons)

        self.radio_calibrated = QRadioButton("Mostrar Calibrada")
        self.radio_calibrated.setChecked(True)
        self.radio_calibrated.toggled.connect(self.showImgSelection)
        layout_radio_buttons.addWidget(self.radio_calibrated)

        self.radio_original = QRadioButton("Mostrar Original")
        self.radio_original.setChecked(False)
        self.radio_original.toggled.connect(self.showImgSelection)
        layout_radio_buttons.addWidget(self.radio_original)

        self.radio_buttons.hide()
        layout.addWidget(self.radio_buttons)

        # ------------ Cancel Button ------------
        line2 = QFrame()
        line2.setFrameShape(QFrame.NoFrame)
        line2.setFixedHeight(2)
        layout.addWidget(line2)

        self.button_finish = QPushButton("Cancelar")
        self.final_status_calibration = False
        self.button_finish.clicked.connect(self.finishCalibrationClicked)
        self.button_finish.setFixedHeight(40)

        layout.addWidget(self.button_finish)

        return widget

    def takePhotoOperationClicked(self) -> None:
        print("Botón presionado")

    def takePhotoCalibrationClicked(self) -> None:
        self.group_box_params.show()

    def showImgSelection(self) -> None:
        if self.radio_calibrated.isChecked():
            self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(self.calibration.getDrawImage()))
        elif self.radio_original.isChecked():
            self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(self.calibration.getRawImage()))
    
    def selectPhotoCalibrationClicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "resources/calibration", "Archivos de Imagen (*.npy *.png)"
        )
        #"Seleccionar Imagen", "resources/calibration", "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp)"
        if path:
            self.calibration.loadRawImage(path)
            self.viewer.loadScene()
            self.viewer.clearVariables()
            self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(self.calibration.getRawImage()))
            self.group_box_params.show()

    def detectColorCheckerClicked(self) -> None:
        self.calibration.detectColorChecker(drawPatches=True)
        self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(self.calibration.getDrawImage()))
        self.group_box_save_apply.show()

    def saveAndApplyClicked(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
                                    self,
                                    "Guardar archivo como",
                                    "resources/calibration/params_sin_titulo.pickle",
                                    "Archivos PICKLE (*.pickle)"
                                )
        if file_path:
            self.final_status_calibration = True
            # Update config json file
            self.config_manager.setValue("calibration", "calibration_params_path", file_path)
            # Save calibration params
            self.calibration.setPathcalibrationParams(file_path)
            self.calibration.saveCalibrationParams()
            # Apply calibration and show it
            model = self.calibration.reconstructModelFromParams()
            img = self.calibration.applyColorCorrection(self.calibration.getRawImage(), model)
            self.calibration.setDrawImage(img)
            self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(img))
            # Update GUI
            self.radio_buttons.show()
            self.button_finish.setText("Finalizar")
            
    def finishCalibrationClicked(self) -> None:
        if self.final_status_calibration:
            msg = "La calibración de color se ha aplicado correctamente."
        else:
            msg = "La calibración de color ha sido cancelada por el usuario."

        QMessageBox.information(
                        self,
                        "Resultado calibración",
                        msg
                    )
        if self.viewer.scene:
            self.viewer.scene.clear()
        self.calibration.clearAllCalibration()
        self.restoreOperationWidget()
    
    def showMask(self, state) -> None:
        if state == 2:
            self.viewer.showMask(True)
        else:
            self.viewer.showMask(False)

    def showPoints(self, state) -> None:
        if state == 2:
            self.viewer.showAllPoints(True)
        else:
            self.viewer.showAllPoints(False)
    
    def updateColorDisplay(self) -> None:
        colors: dict =  self.processing.getTopColors()
        """Actualiza el widget de visualización de color"""
        if colors is not None:
            self.checkbox_show_mask.setChecked(True)
            self.checkbox_show_points.setChecked(True)
            # Color 1
            r = int(colors["Color 1"]["RGB"][0])
            g = int(colors["Color 1"]["RGB"][1])
            b = int(colors["Color 1"]["RGB"][2])
            color_display_button_style: str = f"""
                    font-weight: bold;
                    font-size: 24px;
                    padding: 5px;
                    margin: 0px;
                    border: 1px solid #707070;
                    border-radius: 7px;
                    background-color: rgb({r}, {g}, {b});
                    color: #e0e0e0;
            """
            self.color_display_1.setStyleSheet(color_display_button_style)
            self.group_box_color_1.setTitle(colors["Color 1"]["Pantone Name"])
            l = colors["Color 1"]["LAB"][0]
            a = colors["Color 1"]["LAB"][1]
            b = colors["Color 1"]["LAB"][2]
            self.color_lab_1.setText(f"LAB: ({l:.5f}, {a:.1f}, {b:.1f})")
            delta = colors["Color 1"]["ΔE00"]
            if delta < 0:
                self.color_delta_1.hide()
            else:
                self.color_delta_1.setText(f"ΔE00: {delta:.5f}")
                self.color_delta_1.show()
            # Color 2
            r = int(colors["Color 2"]["RGB"][0])
            g = int(colors["Color 2"]["RGB"][1])
            b = int(colors["Color 2"]["RGB"][2])
            color_display_button_style: str = f"""
                    font-weight: bold;
                    font-size: 24px;
                    padding: 5px;
                    margin: 0px;
                    border: 1px solid #707070;
                    border-radius: 7px;
                    background-color: rgb({r}, {g}, {b});
                    color: #e0e0e0;
            """
            self.color_display_2.setStyleSheet(color_display_button_style)
            self.group_box_color_2.setTitle(colors["Color 2"]["Pantone Name"])
            l = colors["Color 2"]["LAB"][0]
            a = colors["Color 2"]["LAB"][1]
            b = colors["Color 2"]["LAB"][2]
            self.color_lab_2.setText(f"LAB: ({l:.5f}, {a:.1f}, {b:.1f})")
            delta = colors["Color 2"]["ΔE00"]
            if delta < 0:
                self.color_delta_2.hide()
            else:
                self.color_delta_2.setText(f"ΔE00: {delta:.5f}")
                self.color_delta_2.show()
            # Color 3
            r = int(colors["Color 3"]["RGB"][0])
            g = int(colors["Color 3"]["RGB"][1])
            b = int(colors["Color 3"]["RGB"][2])
            color_display_button_style: str = f"""
                    font-weight: bold;
                    font-size: 24px;
                    padding: 5px;
                    margin: 0px;
                    border: 1px solid #707070;
                    border-radius: 7px;
                    background-color: rgb({r}, {g}, {b});
                    color: #e0e0e0;
            """
            self.color_display_3.setStyleSheet(color_display_button_style)
            self.group_box_color_3.setTitle(colors["Color 3"]["Pantone Name"])
            l = colors["Color 3"]["LAB"][0]
            a = colors["Color 3"]["LAB"][1]
            b = colors["Color 3"]["LAB"][2]
            self.color_lab_3.setText(f"LAB: ({l:.5f}, {a:.1f}, {b:.1f})")
            delta = colors["Color 3"]["ΔE00"]
            if delta < 0:
                self.color_delta_3.hide()
            else:
                self.color_delta_3.setText(f"ΔE00: {delta:.5f}")
                self.color_delta_3.show()
    
    def saveRecord(self) -> None:
        id = self.input_id.text().strip()
        if not id:
            QMessageBox.warning(self, "Campo vacío", "Por favor ingrese un identificador antes de guardar.")
            return None
        
        colors: dict = self.processing.getTopColors()
        config = self.config_manager.getConfig()
        results_folder = config['export']['results']['folder_path']

        # Crear carpetas si no existen
        os.makedirs(results_folder, exist_ok=True)
        images_folder = os.path.join(results_folder, "images")
        os.makedirs(images_folder, exist_ok=True)

        # Rutas usando la carpeta configurada
        img_filename_original: str = f"{id.lower().replace(' ','_')}.png"
        img_filename_calibrated: str = f"{id.lower().replace(' ','_')}_calibrated.png"
        excel_path = os.path.join(results_folder, "registros.xlsx")

        date: datetime = datetime.now()


        new_data: dict = {
                            "Identificador": [id],
                            "Fecha Registro": date.strftime("%Y-%m-%d %H:%M:%S"),
                            "Ubicación Imagen": [images_folder],
                            "Método Color": config['color']['method']['selected_method']
                        }

        for i in range(1, 4):
            l: str = f"{colors[f"Color {i}"]["LAB"][0]}"
            a: str = f"{colors[f"Color {i}"]["LAB"][1]}"
            b: str = f"{colors[f"Color {i}"]["LAB"][2]}"
            new_data[f"Pantone {i}"] = f"{colors[f"Color {i}"]["Pantone Name"]}"
            new_data[f"LAB {i}"] = f"({l}, {a}, {b})"
            delta = colors[f"Color {i}"]["ΔE00"]
            if delta < 0:
                delta: str = "NA"
            else:
                delta: str = f"{delta:.5f}"
            new_data[f"ΔE00 {i}"] = f"{delta}"

        new_df = pd.DataFrame(new_data)

        try:
            if os.path.exists(excel_path):
                # Leer el archivo existente
                existing_df = pd.read_excel(excel_path)
                
                # Concatenar los datos existentes con los nuevos
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
                final_df = final_df.astype(str)
            else:
                # Si no existe, usar solo los nuevos datos
                final_df = new_df
            
            # Guardar el DataFrame en Excel
            final_df.to_excel(excel_path, index=False, engine='openpyxl')

            img_path: str = os.path.join(images_folder, img_filename_calibrated)
            self.processing.saveCalibratedImage(img_path)
            msg:str = f"Datos guardados correctamente:\n- Excel: {excel_path}\n- Imagen calibrada: {img_path}"

            if config['export']['results']['save_original_img']:
                img_path: str = os.path.join(images_folder, img_filename_original)
                self.processing.saveOriginalImage(img_path)
                msg = msg + f"\n- Imagen original: {img_path}"

            # Mostrar mensaje de éxito
            QMessageBox.information(self, "Completado", msg)
            
            # Limpiar el campo de texto después de guardar
            self.input_id.clear()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                            f"Error al guardar los datos: {str(e)}")

    def saveImage(self) -> None:
        #if self.source_img_path is None:
        if self.viewer.scene is None:
            QMessageBox.warning(self, "Sin imagen", 
                            "No hay ninguna imagen cargada para guardar.")
            return None

        file_path, _ = QFileDialog.getSaveFileName(
                                    self,
                                    "Guardar imagen como",
                                    "imagen_sin_titulo.png",
                                    "Imágenes PNG (*.png);;Imágenes NPY (*.npy)"
                                )

        if not file_path:
            return None

        try:
            img = self.viewer.getImageArrayFromScene()
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".png":
                cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            elif ext == ".npy":
                np.save(file_path, img)
            else:
                QMessageBox.warning(self, "Formato no soportado",
                                    f"El formato {ext} no es válido para guardar.")
            


            QMessageBox.information(self, "Completado",
                                f"Imagen guardada correctamente en:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                            f"Error al guardar la imagen: {str(e)}")

    def startColorCalibration(self) -> None:
        if self.viewer.scene:
            self.viewer.scene.clear()
        self.group_box_others.hide()
        calibrationWidget = self.createSideCalibrationWidget()
        self.side_panel.addWidget(calibrationWidget)
        self.side_panel.setCurrentWidget(calibrationWidget)

    def restoreOperationWidget(self) -> None:
        current_widget = self.side_panel.currentWidget()
        self.side_panel.setCurrentWidget(self.operationWidget)

        if current_widget != self.operationWidget:
            self.side_panel.removeWidget(current_widget)
            current_widget.deleteLater()

    def selectColorCheckerimage(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "resources/calibration", "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            #self.calibration.setColorCheckerPath(path)
            QMessageBox.information(self, 
                                    "Nuevo Color Checker seleccionado", 
                                    f"Crea nuevamente los parámetros de calibración.")

    def openImage(self) -> None:
        self.source_img_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "resources/test_images", "Archivos de Imagen (*.png *.npy *.dng)"
        )
        if self.source_img_path:

            ext = os.path.splitext(self.source_img_path)[1].lower()
            if ext == ".png":
                image = cv2.cvtColor(cv2.imread(self.source_img_path), cv2.COLOR_BGR2RGB)
            elif ext == ".npy":
                image = np.load(self.source_img_path)
            else:
                QMessageBox.warning(self, "Formato no soportado",
                                    f"El formato {ext} no es válido para abrir.")

            self.calibration.clearAllCalibration()
            self.restoreOperationWidget()

            self.processing.loadImage(image, self.calibration)
            self.viewer.loadScene()
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

                
                if color_method == "Mediana":
                    self.processing.estimateColorsByWeightedMedian(min_weight_threshold=config['color']['method']['threshold_median'])
                elif color_method == "K-Means":
                    self.processing.estimateColorByKMeans(n_clusters=config['color']['method']['number_clusters'],
                                                        min_weight_threshold=config['color']['method']['threshold_kmeans'])
                elif color_method == "Soft Voting":
                    self.processing.estimateColorBySoftVoting(sigma=config['color']['method']['sigma'],
                                                            n_clusters=config['color']['method']['n_clusters_soft'],
                                                            min_weight_threshold=config['color']['method']['threshold_softvoting'])
                self.updateColorDisplay()
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec() #Start the event loop
import sys
import os
import time
from datetime import datetime
import gc
from PySide6.QtCore import (Qt, QRectF, QTimer)
from PySide6.QtGui import (QPixmap, QImage, QPainter, QColor, QPalette)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QGraphicsScene, 
                                QGraphicsView, QFileDialog, QColorDialog, QGraphicsPixmapItem, QGraphicsEllipseItem, 
                                QPushButton, QMessageBox, QFrame, QCheckBox, QRadioButton, QGroupBox, QLabel, QLineEdit, QTabWidget,
                                QSpinBox, QDoubleSpinBox, QFormLayout, QDialog, QComboBox, QStackedWidget)

from picamera2 import Picamera2
import numpy as np
import pandas as pd
import pickle
import torch
import cv2
from skimage.color import (deltaE_ciede2000, lab2rgb, rgb2lab)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Camera():
    def __init__(self):
        self.controls = {}
        self.picam2 = Picamera2()
        self.capture_config = self.picam2.create_still_configuration(
                                main={"size": (3280, 2464)}
                            )
        self.preview_config_seg = self.picam2.create_preview_configuration(
                                main={"size": (720, 720)},
                                sensor={"output_size": (3280, 2464)}
                            )
        self.preview_config_cal = self.picam2.create_preview_configuration(
                                main={"size": (720, 540)},
                                sensor={"output_size": (3280, 2464)}
                            )

        # Flag para saber si la cámara está en modo preview
        self.is_preview_mode = False
        
    def getIsPreviewFlag(self) -> bool:
        return self.is_preview_mode

    def setControls(self, config):
        self.controls = {}
        self.controls = {
            "AeEnable": config['camera']['AEC']['aec'],
            "AwbEnable": config['camera']['AWB']['awb'],
            "Contrast": config['camera']['tuning']['contrast'],
            "Saturation": config['camera']['tuning']['saturation'],
            "Brightness": config['camera']['tuning']['brightness'],
            "Sharpness": config['camera']['tuning']['sharpness']
        }

        if not config['camera']['AEC']['aec']:
            self.controls["ExposureTime"] = config['camera']['AEC']['exposition_time']
            self.controls["AnalogueGain"] = config['camera']['AEC']['gain']
        
        if config['camera']['AWB']['awb']:
            self.controls["AwbMode"] = config['camera']['AWB']['awb_mode']
        else:
            self.controls["ColourGains"] = (config['camera']['AWB']['red_gain'],
                                            config['camera']['AWB']['blue_gain'])

        self.picam2.set_controls(self.controls)

    def startPreview(self, preview_type='segmentation'):
        """Inicia el modo preview con configuración de baja resolución'"""
        if not self.picam2.started:
            # Configurar para preview de baja resolución
            if preview_type == 'segmentation':
                self.picam2.configure(self.preview_config_seg)
            else:
                self.picam2.configure(self.preview_config_cal)
                
            self.picam2.set_controls(self.controls)
            self.picam2.start()
            self.is_preview_mode = True
    
    def stopPreview(self):
        if self.picam2.started:
            """Detiene el modo preview"""
            self.picam2.stop()
            self.is_preview_mode = False
    
    def getPreviewFrame(self, preview_type='segmentation'):
        """Captura un frame del preview
        preview_type: 'segmentation' (2464x2464 centrado) o 'calibration' (3280x2464)"""

        if not self.picam2.started:
            return None
        
        # Capturar frame de baja resolución
        frame = self.picam2.capture_array()
        
        return frame

    def capture(self, n_captures=10):
        self.picam2.configure(self.capture_config)
        self.picam2.set_controls(self.controls)
        
        # Capturar imagen
        self.picam2.start()
        time.sleep(1)
        first_img = self.picam2.capture_array()
        sum_cum = np.zeros(first_img.shape, dtype=np.float32)
        sum_cum += first_img

        for i in range(1, n_captures):
            img = self.picam2.capture_array()
            sum_cum += img
        
        self.picam2.stop()

        mean_rgb_img = (sum_cum / n_captures).astype(np.uint8)

        return mean_rgb_img


class Documentation():
    def __init__(self) -> None:
        self.dpi = 300
        self.is_segmented_flag = False
        self.list_r = [12, 8, 4, 2]
        self.list_eps = [0.1, 0.2, 0.3, 0.4]
        self.export_folder = "exports"
    
    def getIsSegmentedFlag(self):
        return self.is_segmented_flag
    
    def setIsSegmentedFlag(self, value):
        self.is_segmented_flag = value

    def createGuidedFilterComparisonImage(self, processing):
        rows: int = len(self.list_r)
        columns: int = len(self.list_eps)
        # Crea una figura y una cuadrícula de subgráficos (axes)
        # figsize controla el tamaño final de la imagen en pulgadas
        fig, axes = plt.subplots(rows, columns, figsize=(10, 8))

        for i, r in enumerate(self.list_r):
            for j, eps in enumerate(self.list_eps):
                processing.guidedFilter(r, eps**2)
                
                valid_mask: bool = processing.feathered_mask > 0
                
                if (i == 0) and (j == 0):
                    # Encuentra el bounding box de la región válida
                    valid_rows = np.any(valid_mask, axis=1)
                    valid_cols = np.any(valid_mask, axis=0)
                    rmin, rmax = np.where(valid_rows)[0][[0, -1]]
                    cmin, cmax = np.where(valid_cols)[0][[0, -1]]

                # Recorta la imagen al bounding box
                img = processing.calibrated_image[rmin:rmax+1, cmin:cmax+1]
                mask_crop = processing.feathered_mask[rmin:rmax+1, cmin:cmax+1]

                colored_mask = processing.createColoredMask(mask_crop, processing.getFeatheredMaskColor(), 0.6)

                # Muestra la imagen en el subgráfico correspondiente
                ax = axes[i, j]
                ax.imshow(img)
              

                ax.imshow(colored_mask)

                ax.set_xticks([])
                ax.set_yticks([])

                if i == rows - 1:
                    ax.set_xlabel(f'ε = {eps:.1f}²', fontsize=12, rotation=00, labelpad=7, ha='center', va='center')
                
                if j == 0:
                    ax.set_ylabel(f'r = {r}', fontsize=12, rotation=90, labelpad=7, ha='center', va='center')
        
        os.makedirs(self.export_folder, exist_ok=True)
        img_path = os.path.join(self.export_folder, "filter_comparison.png")
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.savefig(img_path, dpi=self.dpi, bbox_inches='tight')
        return img_path

    @staticmethod
    def getColorMethods(config_manager, processing) -> dict:
        config = config_manager.getConfig()
        results: dict = {}
        start_time: float = 0.0
        duration_time: float = 0.0
        
        # Method 1: Weighted Median
        start_time = time.perf_counter()
        processing.estimateColorsByWeightedMedian(min_weight_threshold=config['color']['method']['threshold_median'])
        duration_time = (time.perf_counter()-start_time) * 1e3
        results['Mediana Pesada'] = processing.getTopColors().copy()
        results['Mediana Pesada']['Tiempo Ejecución (ms)'] = duration_time

        # Method 2: K-Means
        start_time = time.perf_counter()
        processing.estimateColorByKMeans(n_clusters=config['color']['method']['number_clusters'],
                                        min_weight_threshold=config['color']['method']['threshold_kmeans'])
        duration_time = (time.perf_counter()-start_time) * 1e3
        results['K-Means'] = processing.getTopColors().copy()
        results['K-Means']['Tiempo Ejecución (ms)'] = duration_time

        # Method 3: Soft Voting
        start_time = time.perf_counter()
        processing.estimateColorBySoftVoting(sigma=config['color']['method']['sigma'],
                                            n_clusters=config['color']['method']['n_clusters_soft'],
                                            min_weight_threshold=config['color']['method']['threshold_softvoting'])
        duration_time = (time.perf_counter()-start_time) * 1e3
        results['Soft Voting'] = processing.getTopColors().copy()
        results['Soft Voting']['Tiempo Ejecución (ms)'] = duration_time

        return results

    def createColorMethodsComparationImage(self, config_manager, processing) -> str:
        
        results: dict = self.getColorMethods(config_manager, processing)

        # Crear la figura con 3 columnas
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        fig.subplots_adjust(wspace=0.001)
        fig.suptitle('Comparación de Colores Predominantes', fontsize=16, fontweight='bold')
        
        methods: list = list(results.keys())
        
        for idx, method in enumerate(methods):
            ax = axes[idx]
            ax.axis('off')
            
            method_data = results[method]
            duration = method_data['Tiempo Ejecución (ms)']
            color_data = method_data['Color 1']
            
            # Extraer datos
            pantone_name = color_data['Pantone Name']
            lab_values = color_data['LAB']
            rgb_values = color_data['RGB']
            delta_e = color_data['ΔE00']
            
            # Título del método
            ax.text(0.5, 0.95, method, ha='center', va='top', fontsize=13, 
                    fontweight='bold', transform=ax.transAxes)
            
            # Duración
            ax.text(0.5, 0.85, f'Tiempo de Ejecución: {duration:.1f} ms', ha='center', va='top', 
                    fontsize=10, transform=ax.transAxes, style='italic')
            
            # Crear tabla
            table_data: list = [
                ['Pantone', pantone_name],
                ['LAB', f'[{lab_values[0]:.5f}, {lab_values[1]:.2f}, {lab_values[2]:.2f}]'],
                ['ΔE00', f'{delta_e:.2f}' if delta_e != -1 else 'N/A']
            ]
            
            # Dibujar tabla
            table = ax.table(cellText=table_data, cellLoc='left',
                            loc='center', bbox=[0.05, 0.42, 0.9, 0.3],
                            colWidths=[0.3, 0.7])
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            
            # Estilo de la tabla
            for i in range(len(table_data)):
                cell = table[(i, 0)]
                cell.set_facecolor('#E8E8E8')
                cell.set_text_props(weight='bold')
                cell.set_height(0.1)
                cell = table[(i, 1)]
                cell.set_facecolor('white')
                cell.set_height(0.1)
            
            # Tarjeta de color RGB (ocupando las 2 columnas)
            rgb_normalized = rgb_values / 255.0
            color_box = plt.Rectangle((0.05, 0.05), 0.9, 0.32, 
                                    facecolor=rgb_normalized, 
                                    edgecolor='black', linewidth=1,
                                    transform=ax.transAxes)
            ax.add_patch(color_box)
        
        os.makedirs(self.export_folder, exist_ok=True)
        img_path = os.path.join(self.export_folder, "color_comparation.png")
        plt.savefig(img_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        return img_path
    
    def createHistogramImage(self, histograms):
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

        os.makedirs(self.export_folder, exist_ok=True)
        img_path = os.path.join(self.export_folder, "segment_color_histogram.png")
        plt.tight_layout()
        plt.savefig(img_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        return img_path


class Calibration():
    def __init__(self, config) -> None:
        self.params_path = "resources/calibration/default_calibration_params.pickle"
        self.config = config
        self.color_checker_raw_image = None
        self.color_patches_raw = None
        self.color_patches_calibrated = None
        self.img_draw = None
        self.model = None
        self.reference_LABs = np.array([[37.986, 13.555, 14.059],
                                        [65.711, 18.13, 17.81],
                                        [49.927, -4.88, -21.925],
                                        [43.139, -13.095, 21.905],
                                        [55.112, 8.844, -25.399],
                                        [70.719, -33.397, -0.199],
                                        [62.661, 36.067, 57.096],
                                        [40.02, 10.41, -45.964],
                                        [51.124, 48.239, 16.248],
                                        [30.325, 22.976, -21.587],
                                        [72.532, -23.709, 57.255],
                                        [71.941, 19.363, 67.857],
                                        [28.778, 14.179, -50.297],
                                        [55.261, -38.342, 31.37],
                                        [42.101, 53.378, 28.19],
                                        [81.733, 4.039, 79.819],
                                        [51.935, 49.986, -14.574],
                                        [51.038, -28.631, -28.638],
                                        [96.539, -0.425, 1.186],
                                        [81.257, -0.638, -0.335],
                                        [66.766, -0.734, -0.504],
                                        [50.867, -0.153, -0.27],
                                        [35.656, -0.421, -1.231],
                                        [20.461, -0.079, -0.973]])

    def getParamsPath(self) -> str:
        return self.params_path
    
    def loadRawImage(self, img) -> None:
        self.color_checker_raw_image = img
        
    def getRawImage(self):
        return self.color_checker_raw_image

    def setDrawImage(self, img) -> None:
        self.img_draw = img

    def getDrawImage(self):
        return self.img_draw
            
    def detectColorChecker(self, drawPatches: bool = False, detectCalibratedImg: bool = False, saveCalibratedPatches: bool = False) -> None:

        if detectCalibratedImg:
            imageBGR = cv2.cvtColor(self.img_draw, cv2.COLOR_RGB2BGR)
        else:
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

            if saveCalibratedPatches:
                self.color_patches_calibrated = chartsRGB[:, 1].copy().reshape(int(width / 3), 1, 3) / 255.0
            else:
                self.color_patches_raw = chartsRGB[:, 1].copy().reshape(int(width / 3), 1, 3) / 255.0

    
    def setPathcalibrationParams(self, file_path) -> None:
        self.params_path = file_path

    def saveCalibrationParams(self) -> None:
        # Save the color patches and configuration to a pickle file
        params = {
            'color_patches': self.color_patches_raw
        }
        with open(self.params_path, 'wb') as f:
            pickle.dump(params, f)
    
    def reconstructModel(self, fromParamsFile: bool = False):
        if fromParamsFile:
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
            self.color_patches_raw = params['color_patches']

        self.model = cv2.ccm_ColorCorrectionModel(self.color_patches_raw, cv2.ccm.COLORCHECKER_Macbeth)
        
        # Configure the model
        self.model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
        self.model.setCCM_TYPE(cv2.ccm.CCM_3x3)
        self.model.setDistance(cv2.ccm.DISTANCE_CIE2000)
        self.model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
        self.model.setLinearGamma(2.2)
        self.model.setLinearDegree(3)
        self.model.setSaturatedThreshold(0, 0.98)
        
        # Run the model
        self.model.run()


    def getMeasuresDeltaE00(self) -> float:
        self.detectColorChecker(drawPatches=False, detectCalibratedImg=True, saveCalibratedPatches=True)
        # Convertir parches detectados a LAB
        patches_rgb = self.color_patches_calibrated.reshape(24, 3)
        detected_lab = rgb2lab(patches_rgb[np.newaxis, :, :]).squeeze()
        delta_e00_values = deltaE_ciede2000(self.reference_LABs, detected_lab)
        measures_delta_e00 = {
                            'mean': np.mean(delta_e00_values),
                            'std dev': np.std(delta_e00_values),
                            'min': np.min(delta_e00_values),
                            'max': np.max(delta_e00_values),
                            'values': delta_e00_values
                            }
        return measures_delta_e00

    def clearAllCalibration(self) -> None:
        self.color_checker_raw_image = None
        self.color_patches_raw = None
        self.img_draw = None
        self.model = None
    
    def applyColorCorrection(self, image):
        # Apply color correction to the image
        image = image.astype(np.float64) / 255.0

        # Perform inference with the model
        calibrated_image = self.model.infer(image)
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
        sam2_checkpoint = "resources/checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        

    # Methods
    def loadImage(self, image, calibration) -> None:
        self.original_image = self.cropSquare(image)
        calibration.reconstructModel(fromParamsFile=True)
        self.calibrated_image = calibration.applyColorCorrection(self.original_image)
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

    def searchPantoneInDatabase(self, pantone_name: str) -> list:
        search = pantone_name.strip().upper()
        matches = np.where(self.pantone_name_colors == search)[0]
        if matches.size > 0:
            idx = matches[0]
        else:
            return None

        return [self.pantone_name_colors[idx], self.pantone_lab_colors[idx]]

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
    def createColoredMask(mask, mask_color, alpha=0.4):
        mask = np.clip(mask, 0, 255)
        color = np.hstack((mask_color/255, [alpha]))
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
    def __init__(self, parent=None, camera=None) -> None:
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
        self.is_calibration = False
        self.camera = camera

        self.setRenderHint(QPainter.Antialiasing) # Smooths the edges of drawn points
        self.setRenderHint(QPainter.SmoothPixmapTransform) # Smooths the image when scaling it
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse) # Zoom pointing to the mouse
        self.setDragMode(QGraphicsView.ScrollHandDrag) # Allows scroll and pan
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # Hides horizontal scroll bar
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # Hides vertical scroll bar
        self.viewport().setCursor(Qt.ArrowCursor) # Changes cursor shape

        # Variables para preview
        self.preview_item = None

    # Methods
    
    def loadScene(self):
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

    def setIsCalibrationFlag(self, value: bool):
        self.is_calibration = value

    def setIsPreviewFlag(self, value: bool):
        self.is_preview = value

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
    
    def clearPreview(self):
        """Limpia el preview"""
        if self.preview_item:
            self.scene.removeItem(self.preview_item)
            self.preview_item = None
    
    def addOverlay(self, pixmap: QPixmap) -> None:
        """
        Añade un QPixmap como una capa superpuesta sobre la imagen principal.
        Si ya existe una capa anterior, la elimina primero.
        """
        # Si ya había una máscara, la eliminamos de la escena
        if self.mask_item:
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
        if not self.camera.getIsPreviewFlag():
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
        if not self.camera.getIsPreviewFlag():
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

    def addPoint(self, x: float, y: float, label: int):
        self.point_coordinates.append((x, y))
        self.point_labels.append(label)
        color = QColor(0, 255, 0) if label == 1 else QColor(255, 0, 0)
        # Dibujamos un círculo rojo para marcar el punto
        radius = (int(self.scene_rect.width())>>8) + self.min_markpoint_radius # Escalar punto a dibujar
        marker = QGraphicsEllipseItem(
            x - radius,
            y - radius,
            radius * 2,
            radius * 2,
        )
        marker.setBrush(color)
        marker.setPen(Qt.NoPen)
        # Aseguramos que el marcador se dibuje encima de la foto
        marker.setZValue(1) 
        self.scene.addItem(marker)
        self.marker_items.append(marker)

    def mouseReleaseEvent(self, event) -> None:
        if (not self.is_calibration) and (not self.camera.getIsPreviewFlag()):
            if event.button() == Qt.LeftButton and not self.is_panning:
                if self.pixmap_item:
                    coordinates = self.mapToScene(event.position().toPoint()).toPoint()
                    # Verificamos si el clic fue DENTRO de la imagen
                    if self.pixmap_item.contains(coordinates):
                        self.addPoint(x=coordinates.x(), y=coordinates.y(), label=1)
                        
            # Reseteamos el estado para el próximo clic
            self.press_pos = None
            self.is_panning = False
        
            # Pasamos el evento a la clase base para que se complete la lógica del drag
            super().mouseReleaseEvent(event)
            self.viewport().setCursor(Qt.ArrowCursor)
    
    def mousePressEvent(self, event) -> None:
        if (not self.is_calibration) and (not self.camera.getIsPreviewFlag()):
            # Solo nos interesa el clic izquierdo p ara iniciar la lógica
            if event.button() == Qt.LeftButton and self.pixmap_item:
                self.click_pos = event.position()
                self.is_panning = False
            elif event.button() == Qt.RightButton and self.pixmap_item:
                coordinates = self.mapToScene(event.position().toPoint()).toPoint()
                # Verificamos si el clic fue DENTRO de la imagen
                if self.pixmap_item.contains(coordinates):
                    self.addPoint(x=coordinates.x(), y=coordinates.y(), label=0)
        
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
        
        # Group Box "Tuning"
        tuning_group = QGroupBox("Tuning")
        tuning_layout = QFormLayout()
        tuning_layout.setHorizontalSpacing(20)
        
        # Contrast (SpinBox)
        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setRange(0.0, 6.0)  # 100µs a 1s
        self.spin_contrast.setValue(0.9)
        self.spin_contrast.setSingleStep(0.1)
        tuning_layout.addRow("Contraste", self.spin_contrast)

        # Saturation (SpinBox)
        self.spin_saturation = QDoubleSpinBox()
        self.spin_saturation.setRange(0.0, 6.0)  # 100µs a 1s
        self.spin_saturation.setValue(0.9)
        self.spin_saturation.setSingleStep(0.1)
        tuning_layout.addRow("Saturación", self.spin_saturation)

        # Brightness (SpinBox)
        self.spin_brightness = QDoubleSpinBox()
        self.spin_brightness.setRange(-1.0, 1.0)  # 100µs a 1s
        self.spin_brightness.setValue(0.0)
        self.spin_brightness.setSingleStep(0.1)
        tuning_layout.addRow("Brillo", self.spin_brightness)

        # Sharpness (SpinBox)
        self.spin_sharpness = QDoubleSpinBox()
        self.spin_sharpness.setRange(0.0, 16.0)  # 100µs a 1s
        self.spin_sharpness.setValue(1.0)
        self.spin_sharpness.setSingleStep(0.1)
        tuning_layout.addRow("Nitidez", self.spin_sharpness)
        
        tuning_group.setLayout(tuning_layout)


        # Group Box "AEC"
        aec_group = QGroupBox("AEC")
        aec_layout = QFormLayout()
        aec_layout.setHorizontalSpacing(20)
        
        # AEC (CheckBox)
        self.check_aec = QCheckBox()
        self.check_aec.setChecked(False)
        self.check_aec.stateChanged.connect(self.aecChecked)
        aec_layout.addRow("Control de exposición automático", self.check_aec)
        
        # Exposition time (SpinBox en microsegundos)
        self.label_exposition_time = QLabel("Tiempo de exposición")
        self.spin_exposition_time = QSpinBox()
        self.spin_exposition_time.setRange(1, 66666)
        self.spin_exposition_time.setValue(9000)
        self.spin_exposition_time.setSuffix(" µs")
        self.spin_exposition_time.setSingleStep(100)
        aec_layout.addRow(self.label_exposition_time, self.spin_exposition_time)
        
        # Gain (SpinBox)
        self.label_gain = QLabel("Ganancia")
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(1.0, 16.0)
        self.spin_gain.setValue(1.0)
        self.spin_gain.setSingleStep(0.1)
        aec_layout.addRow(self.label_gain, self.spin_gain)
        
        aec_group.setLayout(aec_layout)


        # Group Box "AWB"
        awb_group = QGroupBox("AWB")
        awb_layout = QFormLayout()
        awb_layout.setHorizontalSpacing(20)
        
        # AWB (CheckBox)
        self.check_awb = QCheckBox()
        self.check_awb.setChecked(False)
        self.check_awb.stateChanged.connect(self.awbChecked)
        awb_layout.addRow("Balance de blancos automático", self.check_awb)
        
        # AWB Mode (ComboBox)
        self.label_awb_mode = QLabel("Modo AWB")
        self.combo_awb_mode = QComboBox()
        self.combo_awb_mode.addItems([
            "Auto", "Incandescent", "Tungsten", "Fluorescent",
            "Indoor", "Daylight", "Cloudy"
        ])
        self.combo_awb_mode.setCurrentIndex(5)
        self.label_awb_mode.hide()
        self.combo_awb_mode.hide()
        awb_layout.addRow(self.label_awb_mode, self.combo_awb_mode)
        
        # Red Gain (SpinBox)
        self.label_red_gain = QLabel("Ganancia rojo")
        self.spin_red_gain = QDoubleSpinBox()
        self.spin_red_gain.setRange(0.0, 32.0)
        self.spin_red_gain.setValue(1.7)
        self.spin_red_gain.setSingleStep(0.1)
        awb_layout.addRow(self.label_red_gain, self.spin_red_gain)

        # Blue Gain (SpinBox)
        self.label_blue_gain = QLabel("Ganancia azul")
        self.spin_blue_gain = QDoubleSpinBox()
        self.spin_blue_gain.setRange(0.0, 32.0)
        self.spin_blue_gain.setValue(1.4)
        self.spin_blue_gain.setSingleStep(0.1)
        awb_layout.addRow(self.label_blue_gain, self.spin_blue_gain)
        
        awb_group.setLayout(awb_layout)


        layout.addWidget(tuning_group)
        layout.addWidget(aec_group)
        layout.addWidget(awb_group)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab

    def aecChecked(self, state):
        if state == 2:
            self.label_exposition_time.hide()
            self.spin_exposition_time.hide()
            self.label_gain.hide()
            self.spin_gain.hide()
        else:
            self.label_exposition_time.show()
            self.spin_exposition_time.show()
            self.label_gain.show()
            self.spin_gain.show()
    
    def awbChecked(self, state):
        if state == 2:
            self.label_red_gain.hide()
            self.spin_red_gain.hide()
            self.label_blue_gain.hide()
            self.spin_blue_gain.hide()
            self.label_awb_mode.show()
            self.combo_awb_mode.show()
        else:
            self.label_awb_mode.hide()
            self.combo_awb_mode.hide()
            self.label_red_gain.show()
            self.spin_red_gain.show()
            self.label_blue_gain.show()
            self.spin_blue_gain.show()
    
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
        params_layout.setSpacing(20)
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
        filter_layout.addRow("Epsilon (ε)", self.spin_filter_epsilon)
        
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
        pantone_file_layout.setSpacing(20)
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
        self.spin_export_dpi.setRange(50, 600)
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


        # Save images (CheckBox)
        check_save_widget = QWidget()
        check_save_layout = QFormLayout(check_save_widget)
        check_save_layout.setHorizontalSpacing(20)
        check_save_layout.setContentsMargins(0, 0, 0, 0)

        self.check_save_calib_img = QCheckBox()
        self.check_save_calib_img.setChecked(True)
        check_save_layout.addRow("Guardar imagen calibrada", self.check_save_calib_img)

        self.check_save_original_img = QCheckBox()
        self.check_save_original_img.setChecked(True)
        check_save_layout.addRow("Guardar imagen original", self.check_save_original_img)
        results_layout.addWidget(check_save_widget)

        # Folder Resultados
        folder_layout = QHBoxLayout()
        folder_layout.setSpacing(20)
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
        self.spin_contrast.setValue(config['camera']['tuning']['contrast'])
        self.spin_saturation.setValue(config['camera']['tuning']['saturation'])
        self.spin_brightness.setValue(config['camera']['tuning']['brightness'])
        self.spin_sharpness.setValue(config['camera']['tuning']['sharpness'])
        self.check_aec.setChecked(config['camera']['AEC']['aec'])
        self.spin_exposition_time.setValue(config['camera']['AEC']['exposition_time'])
        self.spin_gain.setValue(config['camera']['AEC']['gain'])
        self.check_awb.setChecked(config['camera']['AWB']['awb'])
        self.combo_awb_mode.setCurrentIndex(config['camera']['AWB']['awb_mode'])
        self.spin_red_gain.setValue(config['camera']['AWB']['red_gain'])
        self.spin_blue_gain.setValue(config['camera']['AWB']['blue_gain'])
        
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
        self.check_save_calib_img.setChecked(config['export']['results']['save_calibrated_img'])
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
                                    'tuning':{
                                        'contrast': self.spin_contrast.value(),
                                        'saturation': self.spin_saturation.value(),
                                        'brightness': self.spin_brightness.value(),
                                        'sharpness': self.spin_sharpness.value()
                                    },
                                    'AEC':{
                                        'aec': self.check_aec.isChecked(),
                                        'exposition_time': self.spin_exposition_time.value(),
                                        'gain': self.spin_gain.value()
                                    },
                                    'AWB':{
                                        'awb': self.check_awb.isChecked(),
                                        'awb_mode': self.combo_awb_mode.currentIndex(),
                                        'red_gain': self.spin_red_gain.value(),
                                        'blue_gain': self.spin_blue_gain.value()
                                    }
                                },
                                'calibration': {
                                    'calibration_params_path': self.label_calib_params_path.text()
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
                                    }
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
                                    }
                                },
                                'export': {
                                    'analysis_images': {
                                        'dpi': self.spin_export_dpi.value()
                                    },
                                    'results': {
                                        'folder_path': self.label_results_folder.text(),
                                        'save_calibrated_img': self.check_save_calib_img.isChecked(),
                                        'save_original_img': self.check_save_original_img.isChecked()
                                    }
                                }
                            }
        
        return dict_values


class MainWindow(QMainWindow):
    # Initialization
    def __init__(self) -> None:
        super().__init__()

        self.source_img_path = None
        self.doc = Documentation()

        # Inicializar el gestor de configuración
        self.config_manager = ConfigManager()

        self.camera = Camera()

        self.calibration = Calibration(self.config_manager)
        self.processing = ImageProcessing()

        # Variables para manejo de preview
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.updatePreviewFrame)
        self.preview_fps = 10  # FPS bajo para optimizar RAM
        self.preview_timer.setInterval(int(1000 / self.preview_fps))
        self.current_preview_type = 'segmentation'  # 'segmentation' o 'calibration'
        self.is_preview_active = False

        # Window Settings
        self.setWindowTitle("SACISMC")

        # Widgets
        self.viewer = Viewer(camera=self.camera)
        self.viewer.setFixedSize(720, 720)

        # Panel lateral derecho
        self.side_panel = QStackedWidget()
        self.side_panel.setFixedWidth(460)
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
        action_extract_histogram = submenu_extract.addAction("Histograma de color segmento")
        action_extract_histogram.triggered.connect(self.exportHistogram)
        action_extract_filter_comparison = submenu_extract.addAction("Comparación filtro guiado")
        action_extract_filter_comparison.triggered.connect(self.exportFeatheredComparisonImage)
        action_extract_color_comparison = submenu_extract.addAction("Comparación métodos de color (imagen)")
        action_extract_color_comparison.triggered.connect(self.exportColorComparisonImage)
        action_extract_color_excel = submenu_extract.addAction("Comparación métodos de color (tabla)")
        action_extract_color_excel.triggered.connect(self.exportColorComparison)

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

        # Iniciar preview de segmentación al arrancar
        self.startPreviewSegmentation()

    # Methods
    def startPreviewSegmentation(self):
        """Inicia el preview para el módulo de segmentación"""
        self.current_preview_type = 'segmentation'
        self.camera.startPreview(preview_type='segmentation')

        if self.viewer.scene:
            self.viewer.scene.clear()

        # Cargar escena UNA VEZ
        self.viewer.loadScene()
        self.viewer.clearVariables()

        if not self.preview_timer.isActive():
            self.preview_timer.start()

        self.is_preview_active = True
    
    def startPreviewCalibration(self):
        """Inicia el preview para el módulo de calibración"""
        self.current_preview_type = 'calibration'
        self.camera.startPreview(preview_type='calibration')

        if self.viewer.scene:
            self.viewer.scene.clear()

        self.viewer.loadScene()
        self.viewer.clearVariables()

        if not self.preview_timer.isActive():
            self.preview_timer.start()

        self.is_preview_active = True
    
    def stopPreview(self):
        """Detiene el preview de cámara"""
        if self.preview_timer.isActive():
            self.preview_timer.stop()
        self.camera.stopPreview()
        self.is_preview_active = False
        
        if self.viewer.scene:
            self.viewer.scene.clear()
        
        gc.collect()
    
    def updatePreviewFrame(self):
        """Actualiza el frame del preview en el viewer"""
        if not self.is_preview_active:
            return
        
        frame = self.camera.getPreviewFrame(preview_type=self.current_preview_type)

        pixmap = self.viewer.fromCV2ToQPixmap(frame)

        if self.viewer.scene is None:
            self.viewer.loadScene()
            self.viewer.clearVariables()

        self.viewer.setImageFromPixmap(pixmap)
        del frame

    def openSettingsDialog(self) -> None:
        """Abre el diálogo de configuración"""
        # Detener preview al abrir configuración
        was_preview_active = self.is_preview_active
        preview_type_backup = self.current_preview_type
        if was_preview_active:
            self.stopPreview()

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

            # Reanudar preview con nueva configuración si estaba activo
            if was_preview_active:
                if preview_type_backup == 'segmentation':
                    self.startPreviewSegmentation()
                else:
                    self.startPreviewCalibration()
        else:
            # Si se canceló, solo reanudar preview si estaba activo
            if was_preview_active:
                if preview_type_backup == 'segmentation':
                    self.startPreviewSegmentation()
                else:
                    self.startPreviewCalibration()

    def applyConfigToComponents(self, config):
        """Aplica la configuración a todos los componentes de la aplicación"""
        # Aplicar a Camara
        self.camera.setControls(config)

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

        self.doc.dpi = config['export']['analysis_images']['dpi']
        
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


        line1 = QFrame()
        line1.setFrameShape(QFrame.NoFrame)
        line1.setFixedHeight(1)
        group_others_layout.addWidget(line1)

        # -------------------- REGISTER --------------------

        group_box_register = QGroupBox("Registro")
        group_layout_register = QVBoxLayout()
        group_layout_register.setContentsMargins(13, 15, 13, 10) 
        group_layout_register.setSpacing(10)

        self.input_id = QLineEdit()
        self.input_id.setPlaceholderText("Identificador")
        self.input_id.setFixedHeight(40)
        group_layout_register.addWidget(self.input_id)

        self.id_name = None

        self.log_button = QPushButton("Guardar")
        self.log_button.clicked.connect(self.saveRecord)
        self.log_button.setFixedHeight(50)
        self.log_button.setStyleSheet("font-size: 16px;")
        group_layout_register.addWidget(self.log_button)

        group_box_register.setLayout(group_layout_register)
        group_others_layout.addWidget(group_box_register)


        self.group_box_others.setLayout(group_others_layout)
        self.group_box_others.hide()

        layout.addWidget(self.group_box_others)

        # Botones

        self.return_button = QPushButton("Volver")
        self.return_button.clicked.connect(self.returnClicked)
        self.return_button.setFixedHeight(30)
        self.return_button.setStyleSheet("font-size: 16px; margin: 0px;")
        self.return_button.hide()
        layout.addWidget(self.return_button)

        layout.addStretch()

        return widget
    
    def createSideCalibrationWidget(self) -> QVBoxLayout:
        # Layout total
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

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
        button_take_photo.setFixedHeight(37)
        layout_take_photo.addWidget(button_take_photo)

        layout_load_image.addWidget(group_box_take_photo)

        # ------ o
        label_or = QLabel("-   o   -")
        label_or.setStyleSheet("color: #e0e0e0; font-size: 13px; font-weight: normal; padding: 0px 10px;")
        label_or.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout_load_image.addWidget(label_or)

        # ------ Group Box - Select Photo
        group_box_select_photo = QGroupBox("Seleccionar Foto")
        layout_select_photo = QVBoxLayout(group_box_select_photo)
        layout_select_photo.setContentsMargins(13, 15, 13, 10)
        layout_select_photo.setSpacing(7)

        label_select_photo = QLabel("Desde el explorador de archivos, carga la foto")
        layout_select_photo.addWidget(label_select_photo)

        button_select_photo = QPushButton("Cargar")
        button_select_photo.clicked.connect(self.selectPhotoCalibrationClicked)
        button_select_photo.setFixedHeight(37)
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
        button_detect.setFixedHeight(37)
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

        label_save_apply = QLabel("Aplicar matriz de corrección de color")
        layout_save_apply.addWidget(label_save_apply)

        button_save_apply = QPushButton("Aplicar")
        button_save_apply.clicked.connect(self.applyClicked)
        button_save_apply.setFixedHeight(37)
        layout_save_apply.addWidget(button_save_apply)

        self.group_box_save_apply.hide()
        layout_params.addWidget(self.group_box_save_apply)

        # ------------ Delta Labels ------------
        self.label_measures_delta = QLabel("")
        self.label_measures_delta.setStyleSheet("padding: 0px 0px 0px 2px;")
        self.label_measures_delta.hide()
        layout_params.addWidget(self.label_measures_delta)

        self.deltas: dict = None

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

        # ------------ Finish Buttons ------------
        #line2 = QFrame()
        #line2.setFrameShape(QFrame.NoFrame)
        #line2.setFixedHeight(2)
        #layout.addWidget(line2)

        self.finish_buttons = QWidget()
        layout_finish_buttons = QHBoxLayout(self.finish_buttons)
        layout_finish_buttons.setContentsMargins(0, 0, 0, 0)
        layout_finish_buttons.setSpacing(7)

        self.button_save = QPushButton("Guardar Archivo")
        self.button_save.clicked.connect(self.saveClicked)
        self.button_save.setFixedHeight(50)
        self.button_save.hide()
        layout_finish_buttons.addWidget(self.button_save)

        self.button_cancel = QPushButton("Cancelar")
        self.button_cancel.clicked.connect(self.cancelClicked)
        self.button_cancel.setFixedHeight(50)
        layout_finish_buttons.addWidget(self.button_cancel)

        layout.addWidget(self.finish_buttons)

        return widget
    
    def returnClicked(self) -> None:
        self.doc.setIsSegmentedFlag(False)
        self.startPreviewSegmentation()

        if self.group_box_others.isVisible():
            self.group_box_others.hide()

        if self.return_button.isVisible():
            self.return_button.hide()

    def takePhotoOperationClicked(self) -> None:
        # Detener preview mientras se captura
        self.stopPreview()

        image = self.camera.capture()

        self.calibration.clearAllCalibration()
        self.restoreOperationWidget()

        self.processing.loadImage(image, self.calibration)
        self.viewer.loadScene()
        self.viewer.clearVariables()
        self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(self.processing.getScaledImage()))            
        self.doc.setIsSegmentedFlag(False)
        if self.group_box_others.isVisible():
            self.group_box_others.hide()

        if not self.return_button.isVisible():
            self.return_button.show()

        QMessageBox.information(
                    self,
                    "Completado",
                    "Captura tomada exitosamente!\nAhora crea los point-prompts"
                )

    def takePhotoCalibrationClicked(self) -> None:
        # Detener preview mientras se captura
        self.stopPreview()

        self.calibration.loadRawImage(self.camera.capture())
        self.viewer.loadScene()
        self.viewer.clearVariables()
        self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(self.calibration.getRawImage()))
        self.group_box_params.show()
        self.label_measures_delta.hide()
        self.group_box_save_apply.hide()
        self.radio_buttons.hide()
        self.button_save.hide()

        QMessageBox.information(
                    self,
                    "Completado",
                    "Captura tomada exitosamente!"
                )

    def showImgSelection(self) -> None:
        if self.radio_calibrated.isChecked():
            self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(self.calibration.getDrawImage()))
        elif self.radio_original.isChecked():
            self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(self.calibration.getRawImage()))
    
    def selectPhotoCalibrationClicked(self) -> None:
        # Detener preview al abrir configuración
        was_preview_active = self.is_preview_active
        preview_type_backup = self.current_preview_type
        if was_preview_active:
            self.stopPreview()

        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "resources/calibration", "Archivos de Imagen (*.npy *.png)"
        )
        #"Seleccionar Imagen", "resources/calibration", "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp)"
        if path:
            ext: str = os.path.splitext(path)[1].lower()
            if ext == ".png":
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            elif ext == ".npy":
                img = np.load(path)
            else:
                QMessageBox.warning(self, "Formato no soportado",
                                    f"El formato {ext} no es válido para abrir.")
                return None
            
            self.calibration.loadRawImage(img)
            self.viewer.loadScene()
            self.viewer.clearVariables()
            self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(self.calibration.getRawImage()))
            self.group_box_params.show()
            self.group_box_save_apply.hide()
            self.radio_buttons.hide()
            self.label_measures_delta.hide()
            self.button_save.hide()

    def detectColorCheckerClicked(self) -> None:
        self.calibration.detectColorChecker(drawPatches=True)
        self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(self.calibration.getDrawImage()))

        self.group_box_save_apply.show()

    def applyClicked(self) -> None:
        self.calibration.reconstructModel()
        # Apply calibration and show it
        img = self.calibration.applyColorCorrection(self.calibration.getRawImage())
        self.calibration.setDrawImage(img)
        self.viewer.setImageFromPixmap(self.viewer.fromCV2ToQPixmap(img))
        self.deltas = self.calibration.getMeasuresDeltaE00()
        label: str = f'<span style="text-decoration: overline;">ΔE00</span> : {self.deltas['mean']:.2f}' \
                    f'\u00A0\u00A0|\u00A0\u00A0' \
                    f'σ (ΔE00) : {self.deltas['std dev']:.2f}' \
                    f'\u00A0\u00A0|\u00A0\u00A0' \
                    f'min/max (ΔE00) : {self.deltas['min']:.2f} / {self.deltas['max']:.2f}'
        self.label_measures_delta.setText(label)
        # Update GUI
        self.label_measures_delta.show()
        self.radio_buttons.show()
        self.button_save.show()        

    def saveClicked(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
                                    self,
                                    "Guardar archivo como",
                                    "resources/calibration/params_sin_titulo.pickle",
                                    "Archivos PICKLE (*.pickle)"
                                )
        if file_path:
            # Update config json file
            self.config_manager.setValue("calibration", "calibration_params_path", file_path)
            # Save calibration params
            self.calibration.setPathcalibrationParams(file_path)
            self.calibration.saveCalibrationParams()

            # Create excel report
            excel_path = "exports/calibration_logs.xlsx"

            date: datetime = datetime.now()

            new_data: dict = {
                                "Marca Temporal": date.strftime("%Y-%m-%d %H:%M:%S"),
                                "Archivo Calibración": file_path,
                                "ΔE00 Promedio": f"{self.deltas['mean']:.2f}",
                                "ΔE00 Desviación Estándar": f"{self.deltas['std dev']:.2f}",
                                "ΔE00 Mínimo": f"{self.deltas['min']:.2f}",
                                "ΔE00 Máximo": f"{self.deltas['max']:.2f}",
                                "ΔE00 Parches": f"{self.deltas['values'].tolist()}"
                            }

            new_df = pd.DataFrame([new_data])

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
                time.sleep(1)

                msg = "El archivo de calibración (pickle) se ha guardado correctamente."

                # Mostrar mensaje de éxito
                QMessageBox.information(self, "Completado", msg)

                if self.viewer.scene:
                    self.viewer.scene.clear()

                self.calibration.clearAllCalibration()
                self.restoreOperationWidget()

                # Volver al preview de segmentación
                self.stopPreview()
                self.startPreviewSegmentation()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                f"Error al guardar los datos: {str(e)}")
            
    def cancelClicked(self) -> None:
        msg = "La calibración de color ha sido cancelada por el usuario."

        QMessageBox.information(self, "Cancelado", msg)

        if self.viewer.scene:
            self.viewer.scene.clear()
            
        self.calibration.clearAllCalibration()
        self.restoreOperationWidget()

        # Volver al preview de segmentación
        self.stopPreview()
        self.startPreviewSegmentation()
    
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
    
    def exportColorComparison(self) -> None:
        if self.doc.getIsSegmentedFlag():
            headers = ("Marca Temporal", "Identificador", "Guía - Pantone", "Guía - LAB",
                        "Mediana - Pantone", "Mediana - LAB", "Mediana - ΔE00", "Mediana - Tiempo Ejecución (ms)",
                        "K-Means - Pantone", "K-Means - LAB", "K-Means - ΔE00", "K-Means - Tiempo Ejecución (ms)",
                        "Soft Voting - Pantone", "Soft Voting - LAB", "Soft Voting - ΔE00", "Soft Voting - Tiempo Ejecución (ms)",
                        "Archivo Calibración")
            values = ["NA" for _ in range(len(headers))]
            results: dict = self.doc.getColorMethods(self.config_manager, self.processing)
            config = self.config_manager.getConfig()

            # -------- SET VALUES ---------

            # time stamp
            date: datetime = datetime.now()
            values[0] = date.strftime("%Y-%m-%d %H:%M:%S")

            # median - pantone
            values[4] = results['Mediana Pesada']['Color 1']['Pantone Name']
            # median - lab
            values[5] = results['Mediana Pesada']['Color 1']['LAB'].tolist()
            # median - execution time
            values[7] = f"{results['Mediana Pesada']['Tiempo Ejecución (ms)']:.2f}"

            # k-means - pantone
            values[8] = results['K-Means']['Color 1']['Pantone Name']
            # k-means - lab
            values[9] = results['K-Means']['Color 1']['LAB'].tolist()
            # k-means - execution time
            values[11] = f"{results['K-Means']['Tiempo Ejecución (ms)']:.2f}"

            # soft voting - pantone
            values[12] = results['Soft Voting']['Color 1']['Pantone Name']
            # soft voting - lab
            values[13] = results['Soft Voting']['Color 1']['LAB'].tolist()
            # soft voting - execution time
            values[15] = f"{results['Soft Voting']['Tiempo Ejecución (ms)']:.2f}"

            # calibration file
            values[16] = config['calibration']['calibration_params_path']

            if self.id_name is not None:
                # id
                values[1] = self.id_name
                self.id_name = None

                search = self.processing.searchPantoneInDatabase(values[1].upper())

                if search is not None:
                    # guide - pantone
                    values[2] = search[0]
                    # guide - lab
                    values[3] = search[1].tolist()

                    # median - ΔE00
                    delta = deltaE_ciede2000(np.array(values[5], dtype=np.float32),
                                                np.array(values[3], dtype=np.float32))
                    values[6] = f"{delta:.2f}"
                    # k-means - ΔE00
                    delta = deltaE_ciede2000(np.array(values[9], dtype=np.float32),
                                                np.array(values[3], dtype=np.float32))
                    values[10] = f"{delta:.2f}"
                    # soft voting - ΔE00
                    delta = deltaE_ciede2000(np.array(values[13], dtype=np.float32),
                                                np.array(values[3], dtype=np.float32))
                    values[14] = f"{delta:.2f}"

            # -------- BUILD EXCEL ---------

            os.makedirs(self.doc.export_folder, exist_ok=True)
            excel_path = os.path.join(self.doc.export_folder, "color_methods_comparison.xlsx")
            new_data: dict = {}
            
            for h, v in zip(headers, values):
                new_data[f"{h}"] = f"{v}"
            
            new_df = pd.DataFrame([new_data])
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
                time.sleep(1)

                msg:str = f"Datos guardados correctamente:\n- Excel: {excel_path}"

                # Mostrar mensaje de éxito
                QMessageBox.information(self, "Completado", msg)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                f"Error al guardar los datos: {str(e)}")
        else:
            QMessageBox.warning(self, 
                                "Mascara no encontrada", 
                                "Por favor, correr la segmentación antes de exportar la imagen.")
    
    
    def saveRecord(self) -> None:
        id = self.input_id.text().strip()
        if not id:
            QMessageBox.warning(self, "Campo vacío", "Por favor ingrese un identificador antes de guardar.")
            self.id_name = None
            return None

        self.id_name = id

        colors: dict = self.processing.getTopColors()
        config = self.config_manager.getConfig()
        results_folder = config['export']['results']['folder_path']

        # Crear carpetas si no existen
        os.makedirs(results_folder, exist_ok=True)
        images_folder = os.path.join(results_folder, "images")
        os.makedirs(images_folder, exist_ok=True)

        excel_path = os.path.join(results_folder, "registros.xlsx")
        date: datetime = datetime.now()
        msg:str = f"Datos guardados correctamente:\n- Excel: {excel_path}"

        # Rutas usando la carpeta configurada
        img_filename_original: str = f"{id.lower().replace(' ','_')}.png"
        img_filename_calibrated: str = f"{id.lower().replace(' ','_')}_calibrated.png"

        if config['export']['results']['save_calibrated_img']:
            img_path: str = os.path.join(images_folder, img_filename_calibrated)
            self.processing.saveCalibratedImage(img_path)
            msg = msg + f"\n- Imagen calibrada: {img_path}"

        if config['export']['results']['save_original_img']:
            img_path: str = os.path.join(images_folder, img_filename_original)
            self.processing.saveOriginalImage(img_path)
            msg = msg + f"\n- Imagen original: {img_path}"

        if (not config['export']['results']['save_calibrated_img']) and (not config['export']['results']['save_original_img']):
            images_folder = "NA"

        new_data: dict = {
                            "Identificador": [id],
                            "Fecha Registro": date.strftime("%Y-%m-%d %H:%M:%S"),
                            "Ubicación Imagen": [images_folder],
                            "Archivo Calibración": config['calibration']['calibration_params_path'],
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
        # Detener preview de segmentación
        self.stopPreview()
        self.doc.setIsSegmentedFlag(False)

        self.viewer.setIsCalibrationFlag(True)
        if self.viewer.scene:
            self.viewer.scene.clear()
        self.group_box_others.hide()
        calibrationWidget = self.createSideCalibrationWidget()
        self.side_panel.addWidget(calibrationWidget)
        self.side_panel.setCurrentWidget(calibrationWidget)

        # Iniciar preview de calibración
        self.startPreviewCalibration()

    def restoreOperationWidget(self) -> None:
        self.viewer.setIsCalibrationFlag(False)
        current_widget = self.side_panel.currentWidget()
        self.side_panel.setCurrentWidget(self.operationWidget)

        if current_widget != self.operationWidget:
            self.side_panel.removeWidget(current_widget)
            current_widget.deleteLater()

    def openImage(self) -> None:
        self.source_img_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "resources/test_images", "Archivos de Imagen (*.png *.npy *.dng)"
        )
        if self.source_img_path:
            # Detener preview al abrir imagen desde archivo
            self.stopPreview()

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
            self.doc.setIsSegmentedFlag(False)
            if self.group_box_others.isVisible():
                self.group_box_others.hide()
            
            if not self.return_button.isVisible():
                self.return_button.show()

    def runSegmentation(self) -> None:
        if self.processing:
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
                self.doc.setIsSegmentedFlag(True)
                self.id_name = None
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
        if self.doc.getIsSegmentedFlag():
            img_path = self.doc.createGuidedFilterComparisonImage(self.processing)
            QMessageBox.information(self, 
                                "Imagen exportada exitosamente", 
                                f"Guardada como {img_path}")
        else:
            QMessageBox.warning(self, 
                                "Mascara no encontrada", 
                                "Por favor, correr la segmentación antes de exportar la imagen.")

    def exportColorComparisonImage(self) -> None:
        if self.doc.getIsSegmentedFlag():
            img_path = self.doc.createColorMethodsComparationImage(self.config_manager, self.processing)
            QMessageBox.information(self, 
                                "Imagen exportada exitosamente", 
                                f"Guardada como {img_path}")
        else:
            QMessageBox.warning(self, 
                                "Mascara no encontrada", 
                                "Por favor, correr la segmentación antes de exportar la imagen.")


    def exportHistogram(self) -> None:
        if self.doc.getIsSegmentedFlag():
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
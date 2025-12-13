# Sistema Semi-Automatizado de Identificación de Color para Colonias Microbianas (SACISMC)
 
## Descripción
 
SACISMC es una aplicación de escritorio desarrollada para la identificación y análisis de colores en colonias microbianas mediante técnicas avanzadas de visión por computadora. El sistema utiliza el modelo SAM2 (Segment Anything Model 2) de Meta para la segmentación de colonias y análisis colorimétrico en espacio CIELAB con referencias PANTONE®.
 
## Características Principales
 
- **Captura de Imágenes**: Integración directa con Picamera2 para captura de alta resolución (3280x2464)
- **Segmentación Inteligente**: Utiliza SAM2 de Meta para detección y segmentación de colonias
- **Análisis Colorimétrico**:
  - Análisis en espacio de color CIELAB
  - Cálculo de diferencias de color mediante Delta E (CIEDE2000)
  - Comparación con bases de datos PANTONE®
- **Calibración de Cámara**: Sistema de calibración de color integrado para resultados precisos
- **Exportación de Datos**: Generación de reportes en Excel con análisis detallado
- **Interfaz Gráfica Moderna**: Desarrollada con PySide6 y temas personalizables
 
## Requisitos del Sistema
 
- **Sistema Operativo**: Raspberry Pi OS
- **Hardware**:
  - Raspberry Pi con módulo de cámara compatible con Picamera2
  - Mínimo 4GB de RAM
- **Python**: 3.8 o superior
- **Conexión a Internet**: Necesaria para la instalación inicial
 
## Instalación
 
### 1. Clonar el Repositorio
 
```bash
git clone https://github.com/tu-usuario/Color-Identification-System-in-Microbial-Colonies.git
cd Color-Identification-System-in-Microbial-Colonies
```
 
### 2. Ejecutar el Script de Configuración
 
⚠️ **IMPORTANTE**: Antes del primer uso de la aplicación, es necesario ejecutar el archivo `setup.sh` para instalar todas las dependencias y configurar el entorno:
 
```bash
chmod +x setup.sh
./setup.sh
```
 
El script `setup.sh` realizará las siguientes acciones:
- Creará un entorno virtual de Python con acceso a paquetes del sistema
- Instalará todas las librerías necesarias (numpy, pandas, opencv, torch, PySide6, etc.)
- Descargará los checkpoints del modelo SAM2 (sam2.1_hiera_tiny.pt)
- Creará un acceso directo en el escritorio para facilitar el acceso a la aplicación
 
⏱️ **Nota**: El proceso de instalación puede tomar varios minutos dependiendo de la velocidad de conexión a internet y la capacidad del sistema.
 
## Uso
 
### Iniciar la Aplicación
 
Después de completar la instalación, se puede iniciar la aplicación de dos formas:
 
1. **Desde el acceso directo del escritorio**: Doble clic en el ícono "SACISMC" creado en el escritorio
 
2. **Desde la terminal**:
```bash
.venv/bin/python3 gui_pyside.py
```
 
### Flujo de Trabajo Básico
 
1. **Calibración** (primera vez):
   - Acceder al módulo de calibración
   - Ajustar parámetros de la cámara según sea necesario
   - Capturar imagen de referencia de color
 
2. **Captura y Segmentación**:
   - Capturar imagen de la caja de petri con las colonias
   - Agregar point-prompts sobre la imagen para incluir o excluir regiones
   - Repetir el punto anterior en caso de no obtener la segmentacion deseada
 
3. **Análisis de Color**:
   - Eligir la base de datos PANTONE® de referencia
   - Validar colores obtenidos
 
4. **Exportación**:
   - Exporta resultados a Excel
   - Guarda visualizaciones y reportes
 
## Estructura del Proyecto
 
```
Color-Identification-System-in-Microbial-Colonies/
│
├── gui_pyside.py              # Aplicación principal con interfaz gráfica
├── setup.sh                   # Script de instalación y configuración
├── icon_app.png               # Icono de la aplicación
│
├── resources/
│   ├── calibration/           # Parámetros de calibración
│   ├── config/                # Archivos de configuración
│   ├── pantone_databases/     # Bases de datos de colores PANTONE®
│   ├── styles/                # Temas y estilos de la interfaz
│   └── checkpoints/           # Modelos SAM2 (generado por setup.sh)
│
└── .venv/                     # Entorno virtual (generado por setup.sh)
```
 
## Tecnologías Utilizadas
 
- **PySide6**: Framework de interfaz gráfica
- **PyTorch**: Framework de deep learning
- **SAM2**: Modelo de segmentación de Meta AI
- **OpenCV**: Procesamiento de imágenes
- **scikit-image**: Análisis de color y métricas colorimétricas
- **Picamera2**: Control de cámara Raspberry Pi
- **pandas/openpyxl**: Manejo y exportación de datos
 
 
---
 
**Nota legal**: Pantone® y otras marcas relacionadas son marcas registradas de Pantone LLC.
Las referencias a códigos Pantone® en este proyecto son con fines exclusivamente académicos e ilustrativos, y no implican asociación, patrocinio ni licencia comercial.
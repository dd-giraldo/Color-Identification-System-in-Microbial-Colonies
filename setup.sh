#!/bin/bash

# Usar el directorio donde está el script
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -e "Usando ruta base: $BASE_PATH"


# VENV para picam
python3 -m venv "$BASE_PATH/.venv" --system-site-packages
source "$BASE_PATH/.venv/bin/activate"

echo -e "\nDescargando librerías...\n"

# Install packages
pip install numpy
pip install pandas
pip install openpyxl
pip install matplotlib
pip install opencv-python
pip install opencv-contrib-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install 'git+https://github.com/facebookresearch/sam2.git'
pip install pyside6
pip install scikit-image

# Download checkpoints
mkdir -p "$BASE_PATH/resources/checkpoints/"
wget -nc -P "$BASE_PATH/resources/checkpoints/" https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt


# Create desktop shortcut
DESKTOP_FILE="$HOME/Desktop/SACISMC.desktop"

echo -e "\n\nCreando acceso directo en el escritorio..."

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=SACISMC
Exec=$BASE_PATH/.venv/bin/python3 $BASE_PATH/gui_pyside.py
Path=$BASE_PATH/
Icon=$BASE_PATH/icon_app.png
Terminal=false
Categories=Development;
Name[en_GB]=SACISMC
EOF

# Make it executable
chmod +x "$DESKTOP_FILE"

echo -e "\n¡Acceso directo creado exitosamente!\n"
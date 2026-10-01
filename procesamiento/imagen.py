"""
Preprocesamiento de imágenes para el modelo de Deep Learning
============================================================
Funciones para cargar, redimensionar y normalizar imágenes.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Tamaño de imagen esperado por el modelo
TAMANO_IMAGEN = 224

# Normalización estándar de ImageNet
NORMALIZACION = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def cargar_imagen(ruta: str) -> np.ndarray:
    """
    Carga una imagen desde disco usando OpenCV.

    Args:
        ruta: Ruta al archivo de imagen.

    Returns:
        Imagen como array de NumPy en formato RGB.
    """
    imagen = cv2.imread(ruta)
    if imagen is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta}")
    # OpenCV carga en BGR, convertir a RGB
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    return imagen


def preprocesar_imagen(imagen: np.ndarray) -> torch.Tensor:
    """
    Preprocesa una imagen para el modelo.

    Args:
        imagen: Imagen como array de NumPy (RGB).

    Returns:
        Tensor de PyTorch listo para el modelo.
    """
    transformacion = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((TAMANO_IMAGEN, TAMANO_IMAGEN)),
            transforms.ToTensor(),
            NORMALIZACION,
        ]
    )
    tensor = transformacion(imagen)
    # Agregar dimensión de batch: (1, 3, 224, 224)
    return tensor.unsqueeze(0)


def preprocesar_imagen_pil(imagen_pil: Image.Image) -> torch.Tensor:
    """
    Preprocesa una imagen PIL para el modelo.

    Args:
        imagen_pil: Imagen de Pillow.

    Returns:
        Tensor de PyTorch listo para el modelo.
    """
    transformacion = transforms.Compose(
        [
            transforms.Resize((TAMANO_IMAGEN, TAMANO_IMAGEN)),
            transforms.ToTensor(),
            NORMALIZACION,
        ]
    )
    tensor = transformacion(imagen_pil)
    return tensor.unsqueeze(0)

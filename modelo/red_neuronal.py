"""
Red Neuronal Convolucional (CNN) para clasificación de materiales
================================================================
Modelo simple con PyTorch para reconocer residuos en imágenes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Categorías de materiales que el modelo puede reconocer
CATEGORIAS = [
    "Basura",
    "Cartón",
    "Metal",
    "Papel",
    "Plástico",
    "Vidrio",
]

# Mapeo de nombres de carpetas del dataset a nombres de categoría
NOMBRES_CARPETAS = {
    "basura": "Basura",
    "carton": "Cartón",
    "metal": "Metal",
    "papel": "Papel",
    "plastico": "Plástico",
    "vidrio": "Vidrio",
}

# Tamaño de imagen esperado por el modelo
TAMANO_IMAGEN = 224


class RedNeuronalMateriales(nn.Module):
    """
    CNN simple para clasificación de imágenes de residuos.

    Arquitectura:
        - 3 capas convolucionales con ReLU y MaxPooling
        - 2 capas fully connected
        - Softmax implícito en la salida (CrossEntropyLoss lo aplica)
    """

    def __init__(self, num_clases: int = 7):
        super().__init__()

        # Capas convolucionales
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout para evitar sobreajuste
        self.dropout = nn.Dropout(0.5)

        # Fully connected
        # Después de 3 poolings: 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_clases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pase hacia adelante de la red."""
        # Capa 1: 224x224 -> 112x112
        x = self.pool(F.relu(self.conv1(x)))
        # Capa 2: 112x112 -> 56x56
        x = self.pool(F.relu(self.conv2(x)))
        # Capa 3: 56x56 -> 28x28
        x = self.pool(F.relu(self.conv3(x)))

        # Aplanar
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def crear_modelo(num_clases: int = 7) -> RedNeuronalMateriales:
    """Crea y devuelve una instancia del modelo."""
    return RedNeuronalMateriales(num_clases=num_clases)


def cargar_modelo_entrenado(ruta_pesos: str) -> RedNeuronalMateriales:
    """
    Carga un modelo previamente entrenado.

    Args:
        ruta_pesos: Ruta al archivo .pth con los pesos del modelo.

    Returns:
        Modelo cargado en modo evaluación.
    """
    modelo = crear_modelo(num_clases=len(CATEGORIAS))
    modelo.load_state_dict(torch.load(ruta_pesos, map_location="cpu", weights_only=True))
    modelo.eval()
    return modelo

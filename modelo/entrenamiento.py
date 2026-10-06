"""
Script de entrenamiento del modelo de Deep Learning
===================================================
Entrena la red neuronal con el dataset local y guarda los pesos.
"""

import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modelo.red_neuronal import CATEGORIAS, NOMBRES_CARPETAS, crear_modelo

# === Configuración de entrenamiento ===
RUTA_DATASET = "data"
RUTA_MODELO = "modelo/modelo_entrenado.pth"
RUTA_CLASES = "modelo/clases.json"
EPOCAS = 20
TAMANO_LOTE = 32
TASA_APRENDIZAJE = 0.001
TAMANO_IMAGEN = 224


def obtener_transformaciones():
    """
    Define las transformaciones de preprocesamiento para las imágenes.

    Returns:
        Transformaciones de entrenamiento y validación.
    """
    transformacion_entrenamiento = transforms.Compose(
        [
            transforms.Resize((TAMANO_IMAGEN, TAMANO_IMAGEN)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    transformacion_validacion = transforms.Compose(
        [
            transforms.Resize((TAMANO_IMAGEN, TAMANO_IMAGEN)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transformacion_entrenamiento, transformacion_validacion


def entrenar_epoca(modelo, cargador, criterio, optimizador, dispositivo):
    """Entrena el modelo por una época."""
    modelo.train()
    perdida_total = 0.0
    correctos = 0
    total = 0

    for imagenes, etiquetas in cargador:
        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

        # Pase hacia adelante
        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)

        # Retropropagación
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        # Métricas
        perdida_total += perdida.item()
        _, predichas = torch.max(salidas, 1)
        correctos += (predichas == etiquetas).sum().item()
        total += etiquetas.size(0)

    perdida_promedio = perdida_total / len(cargador)
    precision = 100 * correctos / total
    return perdida_promedio, precision


def validar(modelo, cargador, criterio, dispositivo):
    """Evalúa el modelo en el conjunto de validación."""
    modelo.eval()
    perdida_total = 0.0
    correctos = 0
    total = 0

    with torch.no_grad():
        for imagenes, etiquetas in cargador:
            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)

            perdida_total += perdida.item()
            _, predichas = torch.max(salidas, 1)
            correctos += (predichas == etiquetas).sum().item()
            total += etiquetas.size(0)

    perdida_promedio = perdida_total / len(cargador)
    precision = 100 * correctos / total
    return perdida_promedio, precision


def main():
    """Función principal de entrenamiento."""
    # Verificar si hay GPU disponible
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {dispositivo}")

    # Cargar dataset
    transformacion_entrenamiento, transformacion_validacion = obtener_transformaciones()

    dataset_entrenamiento = datasets.ImageFolder(
        root=f"{RUTA_DATASET}/train",
        transform=transformacion_entrenamiento,
    )
    dataset_validacion = datasets.ImageFolder(
        root=f"{RUTA_DATASET}/val",
        transform=transformacion_validacion,
    )

    cargador_entrenamiento = DataLoader(
        dataset_entrenamiento,
        batch_size=TAMANO_LOTE,
        shuffle=True,
    )
    cargador_validacion = DataLoader(
        dataset_validacion,
        batch_size=TAMANO_LOTE,
        shuffle=False,
    )

    print(f"Clases encontradas: {dataset_entrenamiento.classes}")
    print(f"Imágenes de entrenamiento: {len(dataset_entrenamiento)}")
    print(f"Imágenes de validación: {len(dataset_validacion)}")

    if len(dataset_entrenamiento.classes) != len(CATEGORIAS):
        print(
            f"Advertencia: el dataset tiene {len(dataset_entrenamiento.classes)} clases "
            f"pero CATEGORIAS define {len(CATEGORIAS)}."
        )

    # Guardar las clases (en orden alfabético, como las asigna ImageFolder)
    # para que la interfaz interprete correctamente los índices del modelo
    clases = [NOMBRES_CARPETAS.get(c, c) for c in dataset_entrenamiento.classes]
    with open(RUTA_CLASES, "w", encoding="utf-8") as f:
        json.dump(clases, f, ensure_ascii=False, indent=2)
    print(f"Clases guardadas en {RUTA_CLASES}: {clases}")

    # Crear modelo
    modelo = crear_modelo(num_clases=len(CATEGORIAS)).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)

    # Entrenamiento
    mejor_precision = 0.0

    for epoca in range(1, EPOCAS + 1):
        perdida_ent, precision_ent = entrenar_epoca(
            modelo, cargador_entrenamiento, criterio, optimizador, dispositivo
        )
        perdida_val, precision_val = validar(
            modelo, cargador_validacion, criterio, dispositivo
        )

        print(
            f"Época {epoca}/{EPOCAS} | "
            f"Entrenamiento - Pérdida: {perdida_ent:.4f}, Precisión: {precision_ent:.2f}% | "
            f"Validación - Pérdida: {perdida_val:.4f}, Precisión: {precision_val:.2f}%"
        )

        # Guardar el mejor modelo
        if precision_val > mejor_precision:
            mejor_precision = precision_val
            torch.save(modelo.state_dict(), RUTA_MODELO)
            print(f"  -> Modelo guardado en {RUTA_MODELO}")

    print(f"\nEntrenamiento completado. Mejor precisión: {mejor_precision:.2f}%")


if __name__ == "__main__":
    main()

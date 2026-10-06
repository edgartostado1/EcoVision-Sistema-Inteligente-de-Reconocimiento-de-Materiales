# EcoVision: Sistema Inteligente de Reconocimiento de Materiales

Aplicación en Python que utiliza Deep Learning para reconocer residuos mediante imágenes y recomendar su correcto reciclaje.

## Categorías

- Basura
- Cartón
- Metal
- Papel
- Plástico
- Vidrio

> **Nota:** el dataset actual no incluye imágenes de **Orgánico**. Cuando se
> agreguen, habrá que reentrenar el modelo con `num_clases=7`.

## Tecnologías

- Python
- PyTorch / Torchvision
- OpenCV
- Pillow
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Estructura del proyecto

```
EcoVision/
├── modelo/              # Definición del modelo y entrenamiento
├── procesamiento/       # Preprocesamiento de imágenes
├── interfaz/            # Interfaz de usuario (Streamlit)
├── app.py               # Punto de entrada
├── requirements.txt     # Dependencias
├── data/                # Dataset local (no se sube a GitHub)
└── README.md
```

## Flujo de la aplicación

```
Imagen → Procesamiento → Modelo de Deep Learning → Predicción → Resultado y recomendación
```

## Instalación

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
# Entrenar el modelo (primera vez)
python -m modelo.entrenamiento

# Ejecutar la aplicación
streamlit run app.py
```

## Equipo

Proyecto escolar desarrollado con metodología Scrum.

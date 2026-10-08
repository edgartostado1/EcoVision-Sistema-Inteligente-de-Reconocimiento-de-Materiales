"""
Interfaz de usuario de EcoVision (Streamlit)
============================================
Aplicación web simple para subir imágenes y ver predicciones.
"""

import streamlit as st
from PIL import Image

from modelo.red_neuronal import CATEGORIAS, cargar_modelo_entrenado
from procesamiento.imagen import preprocesar_imagen_pil

# Ruta al modelo entrenado
RUTA_MODELO = "modelo/modelo_entrenado.pth"

# Recomendaciones de reciclaje por categoria
RECOMENDACIONES = {
    "Plástico": "Lava el envase, quítale la tapa y deposítalo en el contenedor amarillo.",
    "Metal": "Aplasta las latas para ahorrar espacio y llévalas al contenedor amarillo.",
    "Vidrio": "Retira las tapas y enjuaga los envases. Deposítalos en el contenedor verde.",
    "Papel": "Dóblalo para ahorrar espacio y colócalo en el contenedor azul.",
    "Cartón": "Aplasta las cajas y deposítalas en el contenedor azul.",
    "basura": "Llévalo al contenedor gris o a un punto limpio.",
}


def cargar_modelo():
    """Carga el modelo entrenado."""
    try:
        return cargar_modelo_entrenado(RUTA_MODELO)
    except FileNotFoundError:
        st.error(
            f"No se encontró el modelo entrenado en '{RUTA_MODELO}'. "
            "Ejecuta primero: python -m modelo.entrenamiento"
        )
        st.stop()


def main():
    """Función principal de la interfaz."""
    st.set_page_config(
        page_title="EcoVision",
        page_icon="♻️",
        layout="centered",
    )

    st.title("EcoVision")
    st.subheader("Sistema Inteligente de Reconocimiento de Materiales")

    st.write(
        "Sube una imagen de un residuo y el modelo te dirá "
        "qué tipo de material es y cómo reciclarlo."
    )

    # Cargar modelo
    modelo = cargar_modelo()

    # Subir imagen
    archivo = st.file_uploader(
        "Selecciona una imagen...",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if archivo is not None:
        # Mostrar imagen
        imagen = Image.open(archivo)
        st.image(imagen, caption="Imagen cargada", use_container_width=True)

        # Predicción
        with st.spinner("Analizando imagen..."):
            tensor = preprocesar_imagen_pil(imagen)

            import torch

            with torch.no_grad():
                salida = modelo(tensor)
                probabilidades = torch.softmax(salida, dim=1)
                indice_predicho = torch.argmax(probabilidades, dim=1).item()
                confianza = probabilidades[0][indice_predicho].item() * 100

        categoria = CATEGORIAS[indice_predicho]

        # Mostrar resultado
        st.success(f"**Predicción: {categoria}** ({confianza:.1f}% de confianza)")

        # Mostrar recomendacion
        st.info(f"**Recomendación:** {RECOMENDACIONES[categoria]}")

        # Mostrar todas las probabilidades
        with st.expander("Ver todas las probabilidades"):
            for i, cat in enumerate(CATEGORIAS):
                prob = probabilidades[0][i].item() * 100
                st.progress(prob / 100)
                st.write(f"{cat}: {prob:.1f}%")


if __name__ == "__main__":
    main()

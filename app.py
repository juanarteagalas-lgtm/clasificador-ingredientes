import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Configuración
# ---------------------------
IMG_SIZE = (224, 224)
CLASES = ['azucar', 'harina', 'huevos', 'mantequilla', 'manzanas']

# ---------------------------
# Información nutricional
# ---------------------------
info_nutricional = {
    "azucar": {
        "calorias": 387,
        "salud": "❌ Poco saludable",
        "comentario": "Alto contenido de azúcar. Consumir con moderación."
    },
    "harina": {
        "calorias": 364,
        "salud": "⚠️ Moderado",
        "comentario": "Fuente de energía, pero refinada."
    },
    "huevos": {
        "calorias": 155,
        "salud": "✅ Saludable",
        "comentario": "Rico en proteínas y nutrientes esenciales."
    },
    "mantequilla": {
        "calorias": 717,
        "salud": "❌ Poco saludable",
        "comentario": "Alta en grasas saturadas."
    },
    "manzanas": {
        "calorias": 52,
        "salud": "✅ Muy saludable",
        "comentario": "Baja en calorías y rica en fibra."
    }
}

# ---------------------------
# Cargar modelo
# ---------------------------
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_ingredientes.h5")

model = cargar_modelo()

# ---------------------------
# Interfaz
# ---------------------------
st.title("🍎 Clasificador de Ingredientes")
st.write("Sube una imagen y el modelo identificará el ingrediente y su información nutricional.")

imagen_subida = st.file_uploader(
    "📷 Sube una imagen (JPG o PNG)",
    type=["jpg", "jpeg", "png"]
)

if imagen_subida is not None:
    imagen = Image.open(imagen_subida).convert("RGB")
    st.image(imagen, caption="Imagen cargada", use_column_width=True)

    # ---------------------------
    # Preprocesamiento
    # ---------------------------
    imagen = imagen.resize(IMG_SIZE)
    img_array = np.array(imagen) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------------------
    # Predicción
    # ---------------------------
    predicciones = model.predict(img_array)
    clase_predicha = np.argmax(predicciones)
    confianza = np.max(predicciones)

    ingrediente = CLASES[clase_predicha]
    info = info_nutricional[ingrediente]

    # ---------------------------
    # Resultados
    # ---------------------------
    st.success(f"🍽️ Ingrediente detectado: **{ingrediente.upper()}**")
    st.info(f"📊 Confianza del modelo: **{confianza:.2%}**")

    st.markdown("### 🧾 Información nutricional")
    st.write(f"🔥 **Calorías (100g):** {info['calorias']} kcal")
    st.write(f"💚 **Salud:** {info['salud']}")
    st.write(f"📝 **Comentario:** {info['comentario']}")


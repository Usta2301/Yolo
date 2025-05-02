import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
import os
from PIL import Image

# Configuración de página
st.set_page_config(
    page_title="Detección de Objetos YOLOv5",
    page_icon="🔍",
    layout="wide"
)

# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

# Sidebar - configuración
st.sidebar.title("🔧 Parámetros de Detección")
conf_thres = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
iou_thres = st.sidebar.slider("IoU mínimo", 0.0, 1.0, 0.45, 0.01)
max_det = st.sidebar.slider("Máx. detecciones", 10, 1000, 300, 10)
show_labels = st.sidebar.checkbox("Mostrar etiquetas", True)
export_csv = st.sidebar.checkbox("Exportar resultados CSV", False)

# Sidebar - subir imagen
uploaded_image = st.sidebar.file_uploader("📷 Subir imagen", type=["jpg", "jpeg", "png"])

# Título
st.title("🔍 Detección de Objetos en Imágenes con YOLOv5")

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Realizar inferencia
    model.conf = conf_thres
    model.iou = iou_thres
    model.max_det = max_det

    with st.spinner("Detectando objetos..."):
        results = model(image)
        df = results.pandas().xyxy[0]

    st.subheader("🖼 Imagen Original")
    st.image(image, channels="BGR", use_column_width=True)

    if not df.empty:
        # Dibujar etiquetas si está activado
        annotated_image = image.copy()
        if show_labels:
            for _, row in df.iterrows():
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                label = f"{row['name']} {row['confidence']:.2f}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Mostrar imagen con anotaciones
        st.subheader("📍 Imagen con Detecciones")
        st.image(annotated_image, channels="BGR", use_column_width=True)

        # Descargar imagen anotada
        retval, buffer = cv2.imencode('.jpg', annotated_image)
        st.download_button("📥 Descargar imagen detectada", buffer.tobytes(), "deteccion.jpg", "image/jpeg")

        # Mostrar tabla de resultados
        st.subheader("📋 Resultados de Detección")
        df_show = df[["name", "confidence"]].groupby("name").agg(["count", "mean"])
        df_show.columns = ["Cantidad", "Confianza Prom."]
        st.dataframe(df_show)

        # Exportar CSV
        if export_csv:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📤 Descargar CSV", csv, "resultados.csv", "text/csv")

        # Gráfico de barras
        st.bar_chart(df['name'].value_counts())
    else:
        st.info("No se detectaron objetos con los parámetros actuales.")
else:
    st.info("Por favor, sube una imagen desde la barra lateral para comenzar.")

# Pie de página
st.markdown("---")
st.caption("Aplicación creada con ❤️ usando Streamlit y YOLOv5 (PyTorch Hub)")

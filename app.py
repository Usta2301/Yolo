import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception:
                st.warning("Intentando método alternativo de carga...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

# Título y descripción
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara o subidas desde tu dispositivo.
""")

# Cargar modelo
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    st.sidebar.title("Parámetros")
    
    with st.sidebar:
        st.subheader('Configuración de detección')
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")
        
        st.subheader('Opciones avanzadas')
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles con esta configuración")
        
    # Método de entrada: cámara o archivo
    st.sidebar.subheader("Fuente de imagen")
    input_method = st.sidebar.radio("Selecciona fuente de imagen:", ["📷 Cámara", "🖼️ Archivo"])

    image = None
    main_container = st.container()

    with main_container:
        if input_method == "📷 Cámara":
            picture = st.camera_input("Captura una imagen con tu cámara")
            if picture:
                bytes_data = picture.getvalue()
                image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                bytes_data = uploaded_file.read()
                image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if image is not None:
            with st.spinner("Detectando objetos..."):
                try:
                    results = model(image)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()

            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Imagen con detecciones")
                    results.render()
                    st.image(image, channels='BGR', use_column_width=True)

                with col2:
                    st.subheader("Objetos detectados")
                    label_names = model.names
                    category_count = {}
                    for category in categories:
                        category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        if category_idx in category_count:
                            category_count[category_idx] += 1
                        else:
                            category_count[category_idx] = 1

                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                        data.append({
                            "Categoría": label,
                            "Cantidad": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })

                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    else:
                        st.info("No se detectaron objetos con los parámetros actuales.")
                        st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
                st.stop()
else:
    st.error("No se pudo cargar el modelo. Por favor verifica las dependencias e inténtalo nuevamente.")
    st.stop()

# Pie de página
st.markdown("---")
st.caption("Desarrollado con Streamlit y PyTorch utilizando YOLOv5.")

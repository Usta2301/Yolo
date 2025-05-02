# ... (importaciones y configuración previas sin cambios)

# Función para agregar etiquetas a la imagen detectada
def draw_labels_on_image(image, predictions, labels, scores, threshold=0.25):
    for i in range(len(predictions)):
        if scores[i] >= threshold:
            box = predictions[i][:4].int()
            label = labels[i]
            conf = scores[i]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Sidebar adicional
with st.sidebar:
    st.subheader("🔍 Opciones de visualización")
    show_labels = st.checkbox("Mostrar etiquetas en imagen", True)
    export_csv = st.checkbox("Exportar detecciones en CSV", False)
    st.subheader("🎨 Tema")
    theme = st.selectbox("Seleccionar tema", ["Claro", "Oscuro"])

    if theme == "Oscuro":
        st.markdown(
            """
            <style>
                body { background-color: #111; color: #fff; }
            </style>
            """,
            unsafe_allow_html=True
        )

# ... (elección de imagen como antes)

if image is not None:
    with st.spinner("Detectando objetos..."):
        results = model(image)

    predictions = results.pred[0]
    boxes = predictions[:, :4].cpu()
    scores = predictions[:, 4].cpu()
    categories = predictions[:, 5].cpu().int()
    label_names = model.names

    labels = [label_names[i] for i in categories]

    # Filtro por clase
    unique_classes = list(set(labels))
    selected_classes = st.multiselect("Filtrar clases detectadas", unique_classes, default=unique_classes)

    # Filtrar predicciones por clase seleccionada
    indices = [i for i, label in enumerate(labels) if label in selected_classes]
    filtered_boxes = boxes[indices]
    filtered_scores = scores[indices]
    filtered_labels = [labels[i] for i in indices]

    if show_labels:
        annotated_img = draw_labels_on_image(image.copy(), filtered_boxes, filtered_labels, filtered_scores)
    else:
        annotated_img = image

    col1, col2 = st.columns(2)

    with col1:
        st.image(annotated_img, channels='BGR', caption="Resultado con anotaciones")

        # Opción de descarga
        retval, buffer = cv2.imencode('.jpg', annotated_img)
        st.download_button("📥 Descargar imagen con anotaciones", buffer.tobytes(), file_name="resultado.jpg", mime="image/jpeg")

    with col2:
        st.subheader("📊 Detalles de detección")
        df_data = []
        for i, label in enumerate(filtered_labels):
            df_data.append({
                "Etiqueta": label,
                "Confianza": f"{filtered_scores[i]:.2f}"
            })

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

        if export_csv:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📤 Exportar CSV", csv, file_name="detecciones.csv", mime="text/csv")

        # Histograma de confianza
        st.bar_chart(df["Confianza"].astype(float))

# Footer
st.markdown("---")
st.caption("Desarrollado por [Tu Nombre] • YOLOv5 + Streamlit • 2025")

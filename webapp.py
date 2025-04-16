import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognition", layout="centered")

@st.cache_resource
def load_my_model():
    return load_model('cnn_mnist_model_new.h5')

model = load_my_model()

st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>✍️ Digit Recognition</h1>
    <p style='text-align: center;'>Draw a digit and click <b>Predict</b></p>
""", unsafe_allow_html=True)

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)", 
    stroke_width=15,
    stroke_color="rgba(0, 0, 0, 1)", 
    background_color="rgba(255, 255, 255, 1)",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    update_streamlit=True, 
)

if canvas_result.image_data is not None:
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = cv2.bitwise_not(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("<div style='height: 42px'></div>", unsafe_allow_html=True)
        if st.button('🔍 Predict', type='primary'):
            with st.spinner('Analyzing...'):
                prediction = model.predict(img, verbose=0)
                predicted_label = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background-color: #e6f4ea; border-radius: 12px;'>
                        <h3 style='margin-bottom: 0;'>🔢 {predicted_label}</h3>
                        <p style='color: gray; margin-top: 4px;'>Predicted Digit</p>
                    </div>
                """, unsafe_allow_html=True)
                st.metric(label="Confidence", value=f"{confidence:.2%}")

    with col2:
        st.markdown("<div style='height: 42px'></div>", unsafe_allow_html=True)
        if st.button('🧹 Clear Drawing'):
            st.rerun()

    with st.expander("🖼 Show processed image"):
        st.image(img[0, :, :, 0], caption="Input to model (28x28)", width=100)






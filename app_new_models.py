import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
import os

# ===========================
# Session state
# ===========================
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# ===========================
# Paths
# ===========================
h5_models = {
    "CNN From Scratch": r"deployment/Best_Model_CNN_From_Scratch.h5",
    "CNN DenseNet169": r"deployment/Best_Model_DenseNet169.h5",
    "CNN ResNet152V2": r"deployment/Best_Model_ResNet152V2_reg.h5",
    "CNN VGG16": r"deployment/VGG16_92.h5",
    "CNN Xception": r"deployment/Best_Model_Xception_reg.h5"
}

tflite_folder = r"deployment/models"
os.makedirs(tflite_folder, exist_ok=True)

# ===========================
# Convert H5 to TFLite if needed
# ===========================
def convert_h5_to_tflite(h5_models_dict, save_folder):
    tflite_paths = {}
    for name, h5_path in h5_models_dict.items():
        tflite_path = os.path.join(save_folder, f"{name.replace(' ', '_').lower()}.tflite")
        if not os.path.exists(tflite_path):
            st.write(f"Converting {name} to TFLite...")
            model = tf.keras.models.load_model(h5_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
        tflite_paths[name] = tflite_path
    return tflite_paths

tflite_models = convert_h5_to_tflite(h5_models, tflite_folder)

# ===========================
# Load TFLite models
# ===========================
def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

loaded_models = {name: load_tflite_model(path) for name, path in tflite_models.items()}

# ===========================
# Class names
# ===========================
class_names = ["Alexgender", "Amitab Bachchan", "Billie Eilish", "Brad Pitt", "Camila Cabello"]

# ===========================
# Helper function to convert PIL image to array
# ===========================
def img_to_array(img):
    return np.array(img, dtype=np.float32)

# ===========================
# Streamlit UI
# ===========================
st.title("Compare CNN Models Predictions")
st.write("Upload Image Of Alexgender, Amitab Bachchan, Billie Eilish, Brad Pitt, Camila Cabello.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess Image
    img_resized = img.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Only show the button if prediction didn't happen yet
    if not st.session_state.predicted:
        if st.button("Predict"):
            st.session_state.predicted = True

    # If prediction is done → show results
    if st.session_state.predicted:
        model_predictions = {}
        all_preds_df = []
        total_probs = np.zeros(len(class_names))

        for model_name, interpreter in loaded_models.items():
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], img_array)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]["index"])[0]

            class_idx = np.argmax(pred)
            predicted_class = class_names[class_idx]

            model_predictions[model_name] = predicted_class
            total_probs += pred

            all_preds_df.append(pd.DataFrame({
                "Class": class_names,
                "Probability": pred,
                "Model": model_name
            }))

        results_df = pd.concat(all_preds_df)

        final_idx = np.argmax(total_probs)
        final_class = class_names[final_idx]

        st.subheader("🎯 Final Ensemble Prediction (Sum of All Models Probabilities)")
        st.markdown(
            f"""
            <h2 style='color: green; font-size: 40px; font-weight: bold;'>
                ✅ {final_class}
            </h2>
            """,
            unsafe_allow_html=True
        )

        final_df = pd.DataFrame({
            "Class": class_names,
            "Summed Probability": total_probs
        })
        st.bar_chart(final_df.set_index("Class")["Summed Probability"])

        st.subheader("📊 Predictions for Each Model")
        for model_name in model_predictions:
            st.write(f"### {model_name} → **{model_predictions[model_name]}**")
            model_df = results_df[results_df["Model"] == model_name]
            st.bar_chart(model_df.set_index("Class")["Probability"])

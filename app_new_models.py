import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd
from huggingface_hub import hf_hub_download

# ===========================
# Session state
# ===========================
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# ===========================
# Load Models from HuggingFace
# ===========================



model_files = {
    "CNN From Scratch": "Best_Model_CNN_From_Scratch.h5",
    "CNN DenseNet169": "Best_Model_DenseNet169.h5",
    "CNN ResNet152V2": "Best_Model_ResNet152V2_reg.h5",
    "CNN VGG16": "VGG16_92.h5",
    "CNN Xception": "Best_Model_Xception_reg.h5"
}

repo_id = "B0o0da/Celebrity-Face-Classification"

loaded_models = {}
for name, filename in model_files.items():
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token="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"  
    )
    loaded_models[name] = load_model(model_path)

# ===========================
# Class names
# ===========================
class_names = ["Alexgender", "Amitab Bachchan", "Billie Eilish", "Brad Pitt", "Camila Cabello"]

# ===========================
# Streamlit UI
# ===========================
st.title("Compare CNN Models Predictions")
st.write("Upload an image of Alexgender, Amitab Bachchan, Billie Eilish, Brad Pitt, or Camila Cabello.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess Image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Only show the button if prediction didn't happen yet
    if not st.session_state.predicted:
        if st.button("Predict"):
            st.session_state.predicted = True  # Hide button after click

    # If prediction is done → show results
    if st.session_state.predicted:

        model_predictions = {}
        all_preds_df = []
        total_probs = np.zeros(len(class_names))

        for model_name, model in loaded_models.items():
            pred = model.predict(img_array)[0]
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

import os
import shutil
from pathlib import Path

import streamlit as st
from PIL import Image
import pipeline
import config

st.set_page_config(page_title="UML Extractor", layout="wide")
st.title("UML Class Diagram Extractor")

model_size_options = {
    "Nano": "n",
    "Medium": "m",
    "X-Large": "x"
}

model_size_default_key = "Medium"
model_size_options_list = list(model_size_options.keys())
model_size_default_index = model_size_options_list.index(model_size_default_key)

# Show radio button with user-facing labels (keys)
selected_model_size_label = st.radio("Select model size:", model_size_options_list, index=model_size_default_index)
selected_model_size_value = model_size_options[selected_model_size_label]

# Upload image
uploaded_file = st.file_uploader("Upload UML diagram image", type=["jpg", "jpeg", "png"])

# Run pipeline only once per upload
if uploaded_file:
    processed_dir = Path(config.PROCESSED_DIR)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    # Save image to file
    image = Image.open(uploaded_file).convert("RGBA")
    image_path = config.PROCESSED_DIR + "/original.png"
    image.save(image_path)
    image_width = image.width

    # Show original image
    st.markdown("### Original")
    st.image(image_path, width=image_width)

    # Run the pipeline
    pipeline.run_uml_extraction_pipeline(image_path, selected_model_size_value, st, uploaded_file.name)
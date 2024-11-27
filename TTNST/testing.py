import os
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image

class TurboTalkStyleTransfer:
    def __init__(self):
        self._configure_environment()
        self._initialize_model()
        self.style_images = self._load_style_images()

    def _configure_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.get_logger().setLevel('ERROR')
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                st.error(f"GPU Configuration Error: {e}")

    def _initialize_model(self):
        try:
            self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        except Exception as model_err:
            st.error(f"Model Loading Failed: {model_err}")
            self.model = None
            st.warning("Please check your internet connection or model availability.")

    def _load_style_images(self):
        # Load images from the local directory
        style_images = {
            "Starry Night": "starry_night.jpg",
            "The Scream": "the_scream.jpg",
            "Mona Lisa Interpretation": "Mona_Lisa.jpeg",
            "Composition VII": "composition_vii.jpg",
            "La Muse": "la_muse.jpg",
            "Abstract Wave": "abstract_wave.png",
            "Abstract Flow": "abstract_flow.jpeg",
            "Abstract Spiral": "abstract_spiral.png",
            "Mosaic": "mosaic.jpg",
            "Oil Painting Essence": "oil_paints.jpg",
            "Watercolor Dream": "watercolors.png",
            "Feathers": "feathers.jpg",
            "Candy": "candy.jpg",
            "Wave": "wave.jpg"
        }
        return style_images

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            st.error(f"Failed to load image from path: {img_path}. Please check the file path.")
            return None, None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]
        img = img.astype(np.float32) / 255.0
        return img[tf.newaxis, :], original_shape

    def apply_style_transfer(self, content_image, style_image):
        stylized_image = self.model(content_image, style_image)[0]
        return stylized_image.numpy()

def main():
    st.title("Artistic Style Transfer")
    st.write("Upload an image and select a style to apply.")

    # Create an instance of the style transfer class
    style_transfer = TurboTalkStyleTransfer()

    # Upload content image
    uploaded_file = st.file_uploader("Choose a content image...", type=["jpg", "jpeg", "png"])
    
    # Select style
    style_name = st.selectbox("Select a style:", list(style_transfer.style_images.keys()))

    if uploaded_file is not None and style_name:
        # Load content image
        content_image, _ = style_transfer.load_image(uploaded_file)

        # Load style image
        style_image_path = style_transfer.style_images[style_name]
        style_image, _ = style_transfer.load_image(style_image_path)

        if content_image is not None and style_image is not None:
            # Apply style transfer
            stylized_image = style_transfer.apply_style_transfer(content_image, style_image)

            # Display images
            st.image(uploaded_file, caption='Content Image', use_column_width=True)
            st.image(style_image[0], caption='Style Image', use_column_width=True)
            st.image(stylized_image[0], caption='Stylized Image', use_column_width=True)

if __name__ == "__main__":
    main()

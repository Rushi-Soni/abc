import os
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import requests
import io
from PIL import Image

class TurboTalkStyleTransfer:
    def __init__(self):
        self._configure_environment()
        self._load_style_resources()
        self._initialize_model()

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

    def _load_style_resources(self):
        self.style_categories = {
            "Classic Art": [
                "Starry Night", "The Scream", "Mona Lisa Interpretation", 
                "Composition VII", "La Muse"
            ],
            "Abstract Styles": [
                "Abstract Wave", "Abstract Flow", "Abstract Spiral", 
                "Mosaic", "Udnie"
            ],
            "Artistic Techniques": [
                "Watercolor Dream", "Oil Painting Essence", 
                "Feathers", "Wave", "Candy"
            ]
        }
        
        self.style_images = {
            "Starry Night": "starry_night.jpg",
            "The Scream": "the_scream.jpg",
            "Mona Lisa Interpretation": "Mona_Lisa.jpeg",
            "Composition VII": "composition_vii.jpg",
            "La Muse": "la_muse.jpg",
            "Abstract Wave": "abstract_wave.png",
            "Abstract Flow": "abstract_flow.jpeg",
            "Abstract Spiral": "abstract_spiral.png",
            "Mosaic": "mosaic.jpg",
            "Udnie": "udnie.jpg",
            "Watercolor Dream": "watercolors.png",
            "Oil Painting Essence": "oil_paints.jpg",
            "Feathers": "feathers.jpg",
            "Wave": "wave.jpg",
            "Candy": "candy.jpg"
        }

    def _initialize_model(self):
        try:
            self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        except Exception as model_err:
            st.error(f"Model Loading Failed: {model_err}")
            self.model = None
            st.warning("Please check your internet connection or model availability.")

    def load_image(self, img_path_or_url):
        try:
            if isinstance(img_path_or_url, str) and img_path_or_url.startswith('http'):
                response = requests.get(img_path_or_url)
                if response.status_code != 200:
                    st.error("Failed to load image from URL. Please check the link.")
                    return None, None
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                if isinstance(img_path_or_url, str):
                    img = cv2.imread(img_path_or_url)
                else:
                    img_array = np.frombuffer(img_path_or_url.read(), np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error("Failed to load image. Please check the file path or URL.")
                return None, None
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_shape = img.shape[:2]
            img = img.astype(np.float32) / 255.0
            return img[tf.newaxis, :], original_shape
        except Exception as e:
            st.error(f"Image Processing Error: {e}")
            return None, None

    def stylize_image(self, content_img, style_img, original_shape, intensity=1.0, enhance_details=False):
        try:
            if enhance_details:
                content_img = tf.image.adjust_contrast(content_img, 1.5)
            
            style_img_resized = tf.image.resize(style_img, [256, 256])
            content_img_resized = tf.image.resize(content_img, [256, 256])
            
            stylized_image = self.model(content_img_resized, style_img_resized)[0]
            
            stylized_image = tf.image.resize(stylized_image, original_shape)
            content_img_original = tf.image.resize(content_img, original_shape)
            
            final_img = content_img_original[0] * (1 - intensity) + stylized_image * intensity
            final_img = tf.clip_by_value(final_img, 0.0, 1.0)
            return final_img.numpy()
        except Exception as e:
            st.error(f"Stylization Error: {e}")
            return None

    def run(self):
        st.title("TurboTalk Style Transfer")
        st.write("Upload an image and select a style to apply.")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        style_option = st.selectbox("Select a style:", list(self.style_images.keys()))
        intensity = st.slider("Style Intensity", 0.0, 1.0, 0.5)
        enhance_details = st.checkbox("Enhance Details")

        if st.button("Apply Style"):
            if uploaded_file is not None:
                content_img, original_shape = self.load_image(uploaded_file)
                if content_img is not None:
                    style_img_path = os.path.join("styles", self.style_images[style_option])
                    style_img, _ = self.load_image(style_img_path)
                    if style_img is not None:
                        final_img = self.stylize_image(content_img, style_img, original_shape, intensity, enhance_details)
                        if final_img is not None:
                            st.image(final_img, caption="Stylized Image", use_column_width=True)
            else:
                st.warning("Please upload an image to apply the style.")

if __name__ == "__main__":
    app = TurboTalkStyleTransfer()
    app.run()

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
            "Starry Night": "images/starry_night.jpg",
            "The Scream": "images/the_scream.jpg",
            "Mona Lisa Interpretation": "images/Mona_Lisa.jpeg",
            "Composition VII": "images/composition_vii.jpg",
            "La Muse": "images/la_muse.jpg",
            "Abstract Wave": "images/abstract_wave.png",
            "Abstract Flow": "images/abstract_flow.jpeg",
            "Abstract Spiral": "images/abstract_spiral.png",
            "Mosaic": "images/mosaic.jpg",
            "Udnie": "images/udnie.jpg",
            "Watercolor Dream": "images/watercolors.png",
            "Oil Painting Essence": "images/oil_paints.jpg",
            "Feathers": "images/feathers.jpg",
            "Wave": "images/wave.jpg",
            "Candy": "images/candy.jpg"
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
            
            # Process images for style transfer
            style_img_resized = tf.image.resize(style_img, [256, 256])
            content_img_resized = tf.image.resize(content_img, [256, 256])
            
            # Apply style transfer
            stylized_image = self.model(content_img_resized, style _img_resized)[0]
            
            # Resize back to original dimensions
            stylized_image = tf.image.resize(stylized_image, original_shape)
            content_img_original = tf.image.resize(content_img, original_shape)
            
            # Apply intensity blending
            final_img = content_img_original[0] * (1 - intensity) + stylized_image * intensity
            final_img = tf.clip_by_value(final_img, 0.0, 1.0)
            final_img = tf.image.convert_image_dtype(final_img, tf.uint8)
            
            return final_img.numpy()
            
        except Exception as e:
            st.error(f"Style Transfer Error: {e}")
            return None

    def add_advanced_styling(self):
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(-45deg, 
                #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient-bg 15s ease infinite;
        }
        @keyframes gradient-bg {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        h1, h2, h3 {
            font-family: 'Montserrat', sans-serif;
            color: white;
            text-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .css-1aumxhk {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(15px);
            border-radius: 15px;
        }
        .stButton>button {
            background-color: #4a4a4a !important;
            color: white !important;
            border-radius: 30px !important;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background-color: #6a4a4a !important;
        }
        </style>
        """, unsafe_allow_html=True)

    def run(self):
        st.set_page_config(
            page_title="TurboTalk Style Transfer", 
            page_icon="🎨", 
            layout="wide"
        )
        
        self.add_advanced_styling()
        
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='font-size: 4rem; color: white;'>
                🎨 TurboTalk Style Transfer
            </h1>
            <p style='color: rgba(255,255,255,0.8); font-size: 1.5rem;'>
                Transform Images Into Masterpieces
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.sidebar:
            st.header("🎨 Style Transfer Settings")
            
            style_category = st.selectbox(
                "Style Category", 
                list(self.style_categories.keys())
            )
            style_selection = st.selectbox(
                "Choose Artistic Style", 
                self.style_categories[style_category]
            )
            
            style_intensity = st.slider(
                "Style Transfer Intensity", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.8, 
                step=0.1
            )
            
            st.subheader("Advanced Options")
            enhance_details = st.checkbox("Enhance Image Details")

        content_image = st.file_uploader(
            "Upload Your Content Image", 
            type=["jpg", "jpeg", "png"],
            help="Select an image to apply artistic style"
        )

        if content_image and st.button("Generate Artistic Masterpiece"):
            with st.spinner("Creating your masterpiece..."):
                content_img, original_shape = self.load_image(content_image)
                style_img, _ = self.load_image(self.style_images[style_selection])
                
                if content_img is not None and style_img is not None:
                    stylized_img = self.stylize_image(
                        content_img, 
                        style_img,
                        original_shape,
                        intensity=style_intensity,
                        enhance_details=enhance_details
                    )
                    
                    if stylized_img is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(content_image, caption="Original Image", use_column_width=True)
                        
                        with col2:
                            processed_img = np.squeeze(stylized_img)
                            st.image(processed_img, caption=f"Stylized with {style_selection}", use_column_width=True)
                        
                        buffered = io.BytesIO ```python
                        Image.fromarray(processed_img).save(buffered, format="PNG")
                        
                        st.download_button(
                            label="💾 Download Masterpiece",
                            data=buffered.getvalue(),
                            file_name="turbotalk_masterpiece.png",
                            mime="image/png"
                        )

def main():
    app = TurboTalkStyleTransfer()
    app.run()

if __name__ == "__main__":
    main()

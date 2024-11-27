import os
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
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

    def load_image(self, img_path_or_url):
        try:
            pil_image = Image.open(img_path_or_url)
            img = np.array(pil_image)
            
            if img is None or img.size == 0:
                st.error("Failed to load image. File might be corrupted.")
                return None, None
            
            if len(img.shape) == 2:  # Grayscale
                img = np.stack((img,) * 3, axis=-1)
            elif img.shape[2] == 4:  # RGBA
                img = img[:,:,:3]
            
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
            final_img = tf.image.convert_image_dtype(final_img, tf.uint8)
            
            return final_img.numpy()
            
        except Exception as e:
            st.error(f"Style Transfer Error: {e}")
            return None

    def run(self):
        st.set_page_config(page_title="TurboTalk Style Transfer", page_icon="🎨", layout="wide")
        
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='font-size: 4rem; color: white;'>
                🎨 TurboTalk Style Transfer
            </h1>
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
                        
                        buffered = io.BytesIO()
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

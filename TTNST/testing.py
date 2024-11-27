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

    def _load_style_resources(self):
        # Define style categories
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
        
        # Use a relative path that works across different environments
        base_path = os.path.join(os.path.dirname(__file__), 'styles')
        
        # If the styles directory doesn't exist, create it
        os.makedirs(base_path, exist_ok=True)
        
        # Create placeholder images if they don't exist
        style_image_data = {
            "starry_night.jpg": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "the_scream.jpg": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "Mona_Lisa.jpeg": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "composition_vii.jpg": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "la_muse.jpg": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "abstract_wave.png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "abstract_flow.jpeg": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "abstract_spiral.png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "mosaic.jpg": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "udnie.jpg": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "watercolors.png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "oil_paints.jpg": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "feathers.jpg": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "wave.jpg": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "candy.jpg": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
        }
        
        # Write placeholder images
        for filename, base64_data in style_image_data.items():
            filepath = os.path.join(base_path, filename)
            if not os.path.exists(filepath):
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(base64_data))
        
        # Update style images dictionary with full paths
        self.style_images = {
            "Starry Night": os.path.join(base_path, "starry_night.jpg"),
            "The Scream": os.path.join(base_path, "the_scream.jpg"),
            "Mona Lisa Interpretation": os.path.join(base_path, "Mona_Lisa.jpeg"),
            "Composition VII": os.path.join(base_path, "composition_vii.jpg"),
            "La Muse": os.path.join(base_path, "la_muse.jpg"),
            "Abstract Wave": os.path.join(base_path, "abstract_wave.png"),
            "Abstract Flow": os.path.join(base_path, "abstract_flow.jpeg"),
            "Abstract Spiral": os.path.join(base_path, "abstract_spiral.png"),
            "Mosaic": os.path.join(base_path, "mosaic.jpg"),
            "Udnie": os.path.join(base_path, "udnie.jpg"),
            "Watercolor Dream": os.path.join(base_path, "watercolors.png"),
            "Oil Painting Essence": os.path.join(base_path, "oil_paints.jpg"),
            "Feathers": os.path.join(base_path, "feathers.jpg"),
            "Wave": os.path.join(base_path, "wave.jpg"),
            "Candy": os.path.join(base_path, "candy.jpg")
        }
        
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

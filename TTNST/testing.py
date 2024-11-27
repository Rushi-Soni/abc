import os
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import tempfile
import time

class NeuralStyleTransferApp:
    def __init__(self):
        # Disable GPU usage
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This ensures CPU is used
        
        # Expanded style images
        self.style_images = {
            "Wave": "wave.jpg",
            "Udnie": "udnie.jpg",
            "The Scream": "the_scream.jpg",
            "Starry Night": "starry_night.jpg",
            "Mosaic": "mosaic.jpg",
            "Abstract Wave": "abstract_wave.png",
            "Abstract Flow": "abstract_flow.jpeg",
            "Abstract Spiral": "abstract_spiral.png",
            "Mona Lisa Interpretation": "Mona_Lisa.jpeg",
            "Watercolor Dream": "watercolors.png",
            "Oil Painting Essence": "oil_paints.jpg",
            "La Muse": "la_muse.jpg",
            "Feathers": "feathers.jpg",
            "Composition VII": "composition_vii.jpg",
            "Candy": "candy.jpg"
        }
        
        # Load the pretrained style transfer model
        try:
            self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        except Exception as model_err:
            st.error(f"Error loading model: {model_err}")
            self.model = None

    def load_image(self, img_path, max_dim=512):
        """Load and preprocess the image."""
        try:
            # Load the image
            img = cv2.imread(img_path)
            if img is None:
                st.error(f"File not found: {img_path}")
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           
            # Resize maintaining aspect ratio
            h, w = img.shape[:2]
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
           
            # Normalize to [0,1] and add batch dimension
            img = img.astype(np.float32) / 255.0
            return img[np.newaxis, ...]
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None

    def stylize_image(self, content_img, style_img):
        """Apply style transfer to the content image using the style image."""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔬 Analyzing Image Characteristics...")
            time.sleep(0.5)
            progress_bar.progress(20)
            
            status_text.text("🎨 Applying Neural Style Transfer Magic...")
            time.sleep(0.5)
            progress_bar.progress(50)
            
            stylized_image = self.model(tf.constant(content_img), tf.constant(style_img))[0]
            
            status_text.text("✨ Finalizing Artistic Transformation...")
            progress_bar.progress(80)
            time.sleep(0.5)
            
            stylized_img_np = np.squeeze(stylized_image.numpy())
            stylized_img_np = (stylized_img_np * 255).astype(np.uint8)
            
            progress_bar.progress(100)
            status_text.text("🎉 Style Transfer Complete!")
            time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            
            return stylized_img_np
        except Exception as e:
            st.error(f"Error during style transfer: {e}")
            return None

    def save_uploaded_file(self, uploaded_file):
        """Save uploaded file to a temporary location."""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                return tmp_file.name
        except Exception as e:
            st.error(f"Error saving uploaded file: {e}")
            return None

    def add_advanced_css(self):
        """Add custom CSS for styling the app."""
        st.markdown(""" 
        <style>
        /* Vibrant Gradient Background */
        .stApp {
            background: linear-gradient(135deg, 
                rgba(102, 126, 234, 0.8), 
                rgba(118, 75, 162, 0.8), 
                rgba(35, 39, 65, 0.8));
            color: white;
        }
        .stButton {
            background-color: rgba(255, 255, 255, 0.2);
            border: none;
            border-radius: 5px;
            color: white;
        }
        .stButton:hover {
            background-color: rgba(255, 255, 255, 0.4);
        }
        </style>
        """, unsafe_allow_html=True)

    def run(self):
        """Run the Streamlit app."""
        st.title("Neural Style Transfer")
        self.add_advanced_css()

        # Upload content image
        content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
        if content_file is not None:
            content_image_path = self.save_uploaded_file(content_file)
            content_image = self.load_image(content_image_path)

            # Select style image
            style_image_name = st.selectbox("Select Style Image", list(self.style_images.keys()))
            style_image_path = self.style_images[style_image_name]
            style_image = self.load_image(style_image_path)

            if st.button("Apply Style Transfer"):
                if content_image is not None and style_image is not None:
                    stylized_image = self.stylize_image(content_image, style_image)
                    if stylized_image is not None:
                        st.image(stylized_image, caption="Stylized Image", use_column_width=True)

if __name__ == "__main__":
    app = NeuralStyleTransferApp()
    app.run()

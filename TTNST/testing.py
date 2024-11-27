import os
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import requests
import matplotlib.pyplot as plt
import tempfile
import time

class NeuralStyleTransferApp:
    def __init__(self):
        # Disable GPU usage
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This ensures CPU is used
        
        # Expanded style images with new abstract and artistic styles
        self.style_images = {
            # Existing Styles
            "Wave": "wave.jpg",
            "Udnie": "udnie.jpg",
            "The Scream": "the_scream.jpg",
            "Starry Night": "starry_night.jpg",
            "Mosaic": "mosaic.jpg",
            
            # New Abstract and Artistic Styles
            "Abstract Wave": "abstract_wave.png",
            "Abstract Flow": "abstract_flow.jpeg",
            "Abstract Spiral": "abstract_spiral.png",
            "Mona Lisa Interpretation": "Mona_Lisa.jpeg",
            "Watercolor Dream": "watercolors.png",
            "Oil Painting Essence": "oil_paints.jpg",
            
            # Additional Existing Styles
            "La Muse": "la_muse.jpg",
            "Feathers": "feathers.jpg",
            "Composition VII": "composition_vii.jpg",
            "Candy": "candy.jpg"
        }
        
        # Limit GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        
        # Load the pretrained style transfer model
        try:
            self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        except Exception as model_err:
            st.error(f"Error loading model: {model_err}")
            self.model = None

    def download_image(self, url):
        """Robust image download with proper headers and error handling"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
           
            # Convert response content to numpy array
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
           
            if img is None:
                raise ValueError("Failed to decode image")
           
            return img
        except Exception as e:
            st.error(f"Error downloading image from {url}: {e}")
            return None

    def load_image(self, img_path_or_url, max_dim=512):
        """Enhanced image loading with robust error handling"""
        try:
            # Checking if the image is a URL
            if img_path_or_url.startswith('http'):
                img = self.download_image(img_path_or_url)
            else:
                # Local file path processing
                if os.path.exists(img_path_or_url):
                    img = cv2.imread(img_path_or_url)
                else:
                    # If file doesn't exist, try loading from the same directory
                    local_path = os.path.join(os.path.dirname(__file__), img_path_or_url)
                    if os.path.exists(local_path):
                        img = cv2.imread(local_path)
                    else:
                        st.error(f"File not found: {img_path_or_url}")
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
        """Enhanced style transfer with progress tracking and error handling"""
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
        """Save uploaded file to a temporary location"""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                return tmp_file.name
        except Exception as e:
            st.error(f"Error saving uploaded file: {e}")
            return None

    def add_advanced_css(self):
        """Comprehensive CSS styling with advanced animations and effects"""
        st.markdown(""" 
        <style>
        /* Vibrant Gradient Background */
        .stApp {
            background: linear-gradient(135deg, 
                rgba(102, 126, 234, 0.8), 
                rgba(118, 75, 162, 0.8), 
                rgba(35, 166, 213, 0.8)
            );
            background-size: 400% 400%;
            animation: gradient-flow 15s ease infinite;
        }

        @keyframes gradient-flow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Modern Typography */
        h1 {
            font-family: 'Montserrat', sans-serif;
            font-weight: 800;
            color: white !important;
            text-shadow: 0 4px 15px rgba(0,0,0,0.3);
            letter-spacing: 1px;
            text-align: center;
        }

        /* Sidebar Enhancements */
        .css-1aumxhk {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        /* Interactive Buttons */
        .stButton>button {
            background-color: #4a4a4a !important;
            color: white !important;
            border-radius: 30px !important;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        }

        .stButton>button:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
            background-color: #6a4a4a !important;
        }

        /* Image Containers */
        .image-container {
            border-radius: 15px;
            overflow: hidden;
            transition: all 0.4s ease;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .image-container:hover {
            transform: scale(1.03);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        }
        </style>
        """, unsafe_allow_html=True)

    def run(self):
        # Set page configuration
        st.set_page_config(
            page_title="Neural Art Generator", 
            page_icon="🎨", 
            layout="wide"
        )

        # Apply advanced CSS
        self.add_advanced_css()
        
        # Animated and dynamic title with more flair
        st.markdown(""" 
        <div style='text-align: center; padding: 20px;'>
            <h1 style='font-size: 3.5rem; color: white; margin-bottom: 10px;'>
                🎨 Neural Artistic Transmutation
            </h1>
            <p style='color: rgba(255,255,255,0.8); font-size: 1.2rem;'>
                Transform your photos into mesmerizing artworks with AI.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload content image
        content_image = st.file_uploader("Upload Your Content Image", type=["jpg", "jpeg", "png"])
        if content_image:
            content_image_path = self.save_uploaded_file(content_image)
            content_img = self.load_image(content_image_path)
            if content_img is None:
                return
        
        # Select style
        style_selection = st.selectbox("Choose Your Artistic Style", list(self.style_images.keys()))
        style_img_path = self.style_images[style_selection]
        style_img = self.load_image(style_img_path)
        if style_img is None:
            return
        
        # Run the style transfer
        if st.button("Generate Art"):
            stylized_img = self.stylize_image(content_img, style_img)
            if stylized_img is not None:
                st.image(stylized_img, caption="Your Artistic Transformation", use_column_width=True)

if __name__ == "__main__":
    app = NeuralStyleTransferApp()
    app.run()

import os
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from icrawler.builtin import GoogleImageCrawler
import random
import io

class TurboTalkStyleTransfer:
    def __init__(self):
        self._configure_environment()
        self._load_style_resources()
        self._initialize_model()

    def _configure_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
        tf.get_logger().setLevel('ERROR')
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                st.error(f"GPU Configuration Error: {e}")

    def _load_style_resources(self):
        # You can expand this by adding more style images
        self.style_images = {
            "Mona Lisa": "images/Mona_Lisa.jpeg",
            "Mosaic": "images/mosaic.jpg",
            "The Scream": "images/the_scream.jpg",
            "Udnie": "images/udnie.jpg"
        }

    def _initialize_model(self):
        try:
            self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        except Exception as model_err:
            st.error(f"Model Loading Failed: {model_err}")
            self.model = None
            st.warning("Please check your internet connection or model availability.")

    def fetch_image_from_google(self, keyword, max_num=1):
        try:
            # Use icrawler to download the image
            google_crawler = GoogleImageCrawler(storage={'root_dir': 'crawler_img'})
            google_crawler.crawl(keyword=keyword, max_num=max_num)
            
            # Dynamically find the path to the downloaded image
            image_path = os.path.join('crawler_img', 'downloads', keyword, '000001.jpg')
            if os.path.exists(image_path):
                st.write(f"Image successfully fetched from Google with keyword '{keyword}': {image_path}")
                
                # Open the image
                img = Image.open(image_path)
                img = img.convert("RGB")  # Ensure it's in RGB format
                img = np.array(img)
                img = img.astype(np.float32) / 255.0
                return img[tf.newaxis, :]  # Add batch dimension
            else:
                st.error(f"No image found at path {image_path}. Please try again.")
                return None
        except Exception as e:
            st.error(f"Error fetching image: {e}")
            return None

    def random_style_image(self):
        # Define chances for each style
        style_weights = {
            "Mona Lisa": 3,
            "Mosaic": 2,
            "The Scream": 1,
            "Udnie": 4
        }
        style_list = list(style_weights.keys())
        style_chances = list(style_weights.values())
        
        # Randomly select a style based on weighted chances
        selected_style = random.choices(style_list, weights=style_chances, k=1)[0]
        return self.style_images[selected_style]

    def stylize_image(self, content_img, style_img, original_shape, intensity=1.0, enhance_details=False):
        try:
            if enhance_details:
                content_img = tf.image.adjust_contrast(content_img, 1.5)
            
            # Process images for style transfer
            style_img_resized = tf.image.resize(style_img, [256, 256])
            content_img_resized = tf.image.resize(content_img, [256, 256])
            
            # Apply style transfer
            stylized_image = self.model(content_img_resized, style_img_resized)[0]
            
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

    def run(self):
        st.set_page_config(
            page_title="TurboTalk Style Transfer", 
            page_icon="🎨", 
            layout="wide"
        )
        
        st.markdown("""<div style='text-align: center; padding: 20px;'>
                        <h1 style='font-size: 4rem; color: white;'>🎨 TurboTalk Style Transfer</h1>
                        <p style='color: rgba(255,255,255,0.8); font-size: 1.5rem;'>Transform Images Into Masterpieces</p>
                    </div>""", unsafe_allow_html=True)

        prompt = st.text_input("Enter a prompt for Google Images", "")

        with st.sidebar:
            st.header("🎨 Style Transfer Settings")
            style_intensity = st.slider("Style Transfer Intensity", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
            st.subheader("Advanced Options")
            enhance_details = st.checkbox("Enhance Image Details")

        if prompt and st.button("Start Style Transfer"):
            with st.spinner("Creating your masterpiece..."):
                content_img = self.fetch_image_from_google(prompt)
                if content_img is not None:
                    style_img_path = self.random_style_image()
                    style_img, _ = self.load_image(style_img_path)
                    
                    if style_img is not None:
                        original_shape = content_img.shape[1:3]
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
                                st.image(content_img[0], caption="Fetched Image", use_column_width=True)
                            
                            with col2:
                                processed_img = np.squeeze(stylized_img)
                                st.image(processed_img, caption="Stylized Image", use_column_width=True)
                            
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

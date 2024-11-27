import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import requests

class StyleTransferApp:
    def __init__(self):
        # Load the TensorFlow Hub model for style transfer
        self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    def load_image(self, img_path_or_url):
        """Load an image from a URL or local path."""
        if isinstance(img_path_or_url, str) and img_path_or_url.startswith('http'):
            response = requests.get(img_path_or_url)
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(img_path_or_url)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img[tf.newaxis, :]

    def stylize_image(self, content_img, style_img, intensity=1.0, enhance_details=False):
        """Apply style transfer to the content image using the style image."""
        if enhance_details:
            content_img = tf.image.adjust_contrast(content_img, 1.5)

        # Resize images to a higher resolution for better quality
        style_img_resized = tf.image.resize(style_img, [512, 512])  # Higher resolution for style
        content_img_resized = tf.image.resize(content_img, [512, 512])  # Higher resolution for content
        
        # Apply style transfer
        stylized_image = self.model(content_img_resized, style_img_resized)[0]

        # Blending the original content image with the stylized image based on intensity
        final_img = content_img_resized[0] * (1 - intensity) + stylized_image * intensity
        final_img = tf.clip_by_value(final_img, 0.0, 1.0)
        return final_img.numpy()

    def run(self):
        """Run the Streamlit application."""
        st.title("Style Transfer Application")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        style_image_url = st.text_input("Enter a URL for the style image:")
        intensity = st.slider("Style Intensity", 0.0, 1.0, 0.5)
        enhance_details = st.checkbox("Enhance Details")

        if st.button("Apply Style"):
            if uploaded_file is not None and style_image_url:
                content_img = self.load_image(uploaded_file)
                style_img = self.load_image(style_image_url)

                final_img = self.stylize_image(content_img, style_img, intensity, enhance_details)
                st.image(final_img, caption="Stylized Image", use_column_width=True)
            else:
                st.warning("Please upload an image and enter a style image URL.")

if __name__ == "__main__":
    app = StyleTransferApp()
    app.run()

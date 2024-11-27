import os
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import io
from PIL import Image
import base64  # Import base64 module to avoid 'NameError'

class TurboTalkStyleTransfer:
    def __init__(self):
        self._configure_environment()
        self._load_style_resources()
        self._initialize_model()

    def _configure_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logs

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

        # Write placeholder images to disk
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

    def _initialize_model(self):
        try:
            self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        except Exception as model_err:
            st.error(f"Model Loading Failed: {model_err}")
            self.model = None

    def stylize_image(self, content_img, style_img, original_shape, intensity=1.0, enhance_details=False):
        try:
            if enhance_details:
                content_img = tf.image.adjust_contrast(content_img, 1.5)
            
            style_img_resized = tf.image.resize(style_img, [256, 256])
            content_img_resized = tf.image.resize(content_img, [256, 256])
            
            # Apply the style transfer model
            stylized_image = self.model(content_img_resized, style_img_resized)[0]
            
            # Resize the result back to the original shape
            stylized_image = tf.image.resize(stylized_image, [original_shape[0], original_shape[1]])
            return stylized_image.numpy().astype(np.uint8)
        except Exception as err:
            st.error(f"Image stylization failed: {err}")
            return None

# Example usage:
def main():
    app = TurboTalkStyleTransfer()
    app.run()

if __name__ == "__main__":
    main()

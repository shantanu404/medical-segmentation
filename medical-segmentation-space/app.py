import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = load_model("model.keras")

def preprocess_image(image):
    # Convert to grayscale
    img = np.array(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize to 256x256
    img = cv2.resize(img, (256, 256))
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    # Add channel dimension
    img = np.expand_dims(img, axis=-1)
    return img

def segment_image(image):
    img = preprocess_image(image)
    # Model expects batch dimension
    pred = model.predict(np.expand_dims(img, axis=0))[0, :, :, 0]
    # Threshold to get binary mask
    mask = (pred > 0.5).astype(np.uint8) * 255
    # Resize mask back to original image size
    orig_size = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
    mask_resized = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(mask_resized)

demo = gr.Interface(
    fn=segment_image,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=gr.Image(type="pil", label="Segmented Mask"),
    title="Breast Cancer Segmentation",
    description="Upload an ultrasound image. The model will segment the tumor region."
)

if __name__ == "__main__":
    demo.launch()
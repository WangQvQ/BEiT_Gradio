import gradio as gr
from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import requests
import numpy as np

# Load the pre-trained BEiT model and feature extractor
feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-large-patch16-512')
model = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-512')


def classify_image(input_image):
    image = Image.fromarray(input_image.astype('uint8'))
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    return {"Predicted Class": predicted_class}


iface = gr.Interface(
    fn=classify_image,
    inputs=gr.inputs.Image(type="numpy"),  # Specify input type as numpy array
    outputs="json",
    live=True,
    title="BEiT 图像描述",
    description="上传一张图像获取描述结果"
)

if __name__ == "__main__":
    iface.launch()

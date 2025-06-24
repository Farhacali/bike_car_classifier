import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import gdown

# Download model from Google Drive if not already present
os.makedirs("model", exist_ok=True)
model_path = "model/bike_car_model.pth"
file_id = "1F7Uat_TJNUYZNF0YbwdrkqL1-zrg6Bjh"

if not os.path.exists(model_path):
    print("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load ResNet18 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Labels and transforms
labels = ['Bike', 'Car']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction logic like Cat/Dog example
def predict_image(img):
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        confidence = probs[1].item()  # Car confidence

        if confidence > 0.95:
            return "✅ Prediction: It's a Car 🚗"
        elif confidence < 0.05:
            return "✅ Prediction: It's a Bike 🏍️"
        else:
            return "❌ This image doesn't appear to be a clear bike or car."

# Gradio interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🚘 Bike vs Car Image Classifier",
    description="Upload an image and let AI decide whether it's a Bike 🏍️ or a Car 🚗",
    theme="soft"
)

if __name__ == "__main__":
    interface.launch()


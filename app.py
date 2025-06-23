import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import gdown

# 🔽 Download model if not already present
model_path = "model/bike_car_model.pth"
gdrive_url = "https://drive.google.com/uc?id=1F7Uat_TJNUYZNF0YbwdrkqL1-zrg6Bjh"

if not os.path.exists(model_path):
    os.makedirs("model", exist_ok=True)
    print("📥 Downloading model from Google Drive...")
    gdown.download(gdrive_url, model_path, quiet=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 model (NOT ResNet50)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Labels
labels = ['Bike', 'Car']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction function
def classify_image(image):
    img = image.convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        return {labels[i]: float(probs[i]) for i in range(2)}

# Gradio UI
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Bike vs Car Classifier",
    description="Upload an image to classify it as either a bike 🏍️ or a car 🚗",
    theme="soft"
)

# Run
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8080)

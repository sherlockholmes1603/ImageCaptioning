# Main script to run the training.
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from models import CaptioningModel  # Replace with your model import
from utils import clean_sentence  # Replace with your sentence cleaning function
from Dataloader import FlickrDataset  # Replace with your dataset import
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
model_path = "model_checkpoint.pth"  # Path to the trained model checkpoint

# Transform for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model
model = CaptioningModel()  # Adjust to match your model's parameters
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load vocabulary
dataset = FlickrDataset(root_folder="path_to_images", captions_file="path_to_captions.csv", transform=None)
vocab = dataset.vocab

# Predict caption for the given image
def predict_caption(image_tensor):
    with torch.no_grad():
        output = model.generate_caption(image_tensor)  # Define `generate_caption` in your model
        sentence = clean_sentence(output, vocab)
    return sentence

# Endpoint for generating captions
@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    try:
        data = request.json
        image_url = data.get("image_url")
        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400
        
        # Download the image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate the caption
        caption = predict_caption(image_tensor)
        return jsonify({"caption": caption}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

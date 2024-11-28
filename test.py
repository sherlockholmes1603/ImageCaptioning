import torch
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from models import CaptioningModel  # Placeholder for your model
from utils import clean_sentence  # Placeholder for a sentence cleaning function
from Dataloader import FlickrDataset  # Use dataset to access vocab

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
model_path = "model_checkpoint.pth"  # Path to the trained model checkpoint
image_path = "path_to_image.jpg"  # Replace with the actual image path
vocab_path = "path_to_vocab_file.pkl"  # If needed for vocab loading

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
dataset = FlickrDataset(root_folder="path_to_images", captions_file="path_to_captions.csv", transform=None)  # Replace paths
vocab = dataset.vocab

# Load and preprocess the image
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img.to(device)

# Generate caption
def predict_caption(image_tensor):
    with torch.no_grad():
        output = model.generate_caption(image_tensor)  # Define generate_caption in your model
        sentence = clean_sentence(output, vocab)
    return sentence

# Run prediction
if __name__ == "__main__":
    img_tensor = load_image(image_path)
    caption = predict_caption(img_tensor)
    print(f"Predicted Caption: {caption}")

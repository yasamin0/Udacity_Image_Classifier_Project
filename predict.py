import argparse
import json
from math import ceil
from PIL import Image
import torch
import numpy as np
from torchvision import models

def parse_arguments():
    """
    Parses command-line arguments for image prediction.
    """
    parser = argparse.ArgumentParser(description="Image Classifier Prediction Script")
    parser.add_argument('--image', type=str, required=True, help='Path to the image file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the saved model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to JSON file mapping categories to names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available.')

    return parser.parse_args()

def load_model(checkpoint_path):
    """
    Loads a pre-trained model and its parameters from a checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path)

    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

def preprocess_image(image_path):
    """
    Processes an image for input into a PyTorch model.
    """
    image = Image.open(image_path)

    # Resize and crop
    image = image.resize((256, 256))
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Normalize
    image = np.array(image) / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - mean) / std

    # Transpose color channels
    image = image.transpose((2, 0, 1))

    return torch.tensor(image, dtype=torch.float).unsqueeze(0)

def predict(image_tensor, model, device, top_k):
    """
    Predicts the top K classes for an image using the trained model.
    """
    model.to(device)
    image_tensor = image_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)
    top_probs, top_indices = probabilities.topk(top_k, dim=1)

    top_probs = top_probs.cpu().numpy().flatten()
    top_indices = top_indices.cpu().numpy().flatten()

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_probs, top_classes

def display_predictions(probs, classes, category_names):
    """
    Displays the predicted classes and their probabilities.
    """
    print("Top Predictions:")
    for i, (cls, prob) in enumerate(zip(classes, probs), 1):
        print(f"{i}. {category_names.get(cls, 'Unknown')} - {ceil(prob * 100)}%")

def main():
    """
    Main function for the prediction script.
    """
    args = parse_arguments()

    # Load category names
    with open(args.category_names, 'r') as f:
        category_names = json.load(f)

    # Load model
    model = load_model(args.checkpoint)

    # Process image
    image_tensor = preprocess_image(args.image)

    # Set device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Predict
    top_probs, top_classes = predict(image_tensor, model, device, args.top_k)

    # Display results
    display_predictions(top_probs, top_classes, category_names)

if __name__ == '__main__':
    main()

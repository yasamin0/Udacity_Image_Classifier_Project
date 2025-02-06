import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn, optim
import numpy as np
import json
from torchvision import datasets, transforms, models
from collections import OrderedDict
from get_input_args import get_input_args_predict, validate_predict_args

# Pre-trained models available
pretrained_models = {
    'alexnet': models.alexnet(pretrained=True),
    'vgg16': models.vgg16(pretrained=True),
}

def load_pretrained_model(model_name='vgg16'):
    """
    Retrieves a pre-trained model based on the specified architecture name.
    """
    try:
        return pretrained_models[model_name]
    except KeyError:
        print(f"Model {model_name} is not available.")
        return None

def map_categories_to_names(file_path='cat_to_name.json'):
    """
    Maps category indices to class names using a JSON file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def load_checkpoint(checkpoint_path):
    """
    Loads a model checkpoint and rebuilds the model with its saved parameters.
    """
    try:
        checkpoint = torch.load(checkpoint_path)
        print(f"Loaded model architecture: {checkpoint['model_name']}")

        model = load_pretrained_model(checkpoint['model_name'])
        if model:
            model.load_state_dict(checkpoint['model_state_dict'])
            model.classifier = checkpoint['classifier']
            model.class_to_idx = checkpoint['class_to_idx']

            for param in model.parameters():
                param.requires_grad = False
            return model
        else:
            raise ValueError("Model architecture not recognized.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def preprocess_image(image_path):
    """
    Prepares an image for input into the model: resizing, cropping, and normalizing.
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    with Image.open(image_path) as image:
        return transform_pipeline(image)

def make_prediction(image_path, model, top_k=5):
    """
    Predicts the top K classes for a given image using a trained model.
    """
    image_tensor = preprocess_image(image_path).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = probabilities.topk(top_k)

    # Convert indices to class labels
    idx_to_class = {idx: cls for cls, idx in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices.squeeze().tolist()]
    return top_probs.squeeze().tolist(), top_classes

def display_image_with_predictions(image, probabilities, classes, class_labels):
    """
    Displays the image along with the top predicted classes and their probabilities.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))
    image_np = image.numpy().transpose((1, 2, 0))
    ax1.imshow(np.clip(image_np, 0, 1))
    ax1.axis('off')

    ax2.barh(range(len(classes)), probabilities)
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels([class_labels.get(cls, "Unknown") for cls in classes])
    ax2.set_xlim(0, 1.1)
    ax1.set_title(class_labels.get(classes[0], "Unknown"))

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to load the model, process the image, and display predictions.
    """
    args = get_input_args_predict()
    validate_predict_args(args)

    print("Loading model...")
    model = load_checkpoint(args.checkpoint)
    if not model:
        print("Model could not be loaded. Exiting...")
        return

    image_path = args.image_dir
    top_k = int(args.top_k) if args.top_k else 5
    probabilities, classes = make_prediction(image_path, model, top_k)

    class_labels = map_categories_to_names()
    display_image_with_predictions(preprocess_image(image_path), probabilities, classes, class_labels)

if __name__ == '__main__':
    main()

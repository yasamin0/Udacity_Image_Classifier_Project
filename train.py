import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import json
from torchvision import datasets, transforms, models
from collections import OrderedDict
from get_input_args import get_input_args_train
from get_input_args import validate_train_args, validate_predict_args

# Pre-trained models available for use
pretrained_models = {
    'alexnet': models.alexnet(pretrained=True),
    'vgg16': models.vgg16(pretrained=True),
}

def load_model_from_checkpoint(filepath):
    """
    Loads a pre-trained model along with its trained weights and parameters.
    Returns the model object or raises an error if loading fails.
    """
    checkpoint = torch.load(filepath)
    try:
        model = select_model_architecture(checkpoint['model_name'])
        print(f"Loading architecture: {checkpoint['model_name']}")
        if model:
            model.classifier = checkpoint['classifier']
            model.load_state_dict(checkpoint['model_state_dict'])
            model.class_to_idx = checkpoint['class_to_idx']

            # Freeze parameters to prevent updates during inference
            for param in model.parameters():
                param.requires_grad = False
        else:
            raise ValueError("Could not load model from checkpoint.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    return model

def select_model_architecture(model_name):
    """
    Select and return a pre-trained model architecture based on the provided name.
    """
    try:
        return pretrained_models[model_name]
    except KeyError as e:
        print(f"Model {model_name} not available. Error: {e}")
        return None

def configure_model_classifier(model, architecture, hidden_units=512):
    """
    Configures the classifier section of the model based on the architecture.
    Allows setting custom hidden layer units.
    """
    for param in model.parameters():
        param.requires_grad = False

    if architecture == 'vgg16':
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    else:
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(9216, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.3)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    model.classifier = classifier
    return model

def save_checkpoint(model, architecture, save_directory, dataset, epochs, optimizer, classifier):
    """
    Saves the trained model as a checkpoint file, including metadata such as
    architecture, classifier, optimizer state, and class-to-index mapping.
    """
    model.class_to_idx = dataset.class_to_idx
    checkpoint = {
        'model_name': architecture,
        'epochs': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classifier': classifier,
        'class_to_idx': model.class_to_idx
    }
    filepath = os.path.join(save_directory, "model_checkpoint.pth")
    torch.save(checkpoint, filepath)

def train_model(model, train_loader, valid_loader, learning_rate=0.001, epochs=5, device='cpu'):
    """
    Handles the training process, including backpropagation and periodic validation.
    Prints metrics like training loss and validation accuracy during training.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    device = torch.device("cuda" if device == 'gpu' else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        validation_loss, accuracy = evaluate_model(model, valid_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {total_loss/len(train_loader):.3f}.. "
              f"Validation loss: {validation_loss/len(valid_loader):.3f}.. "
              f"Validation accuracy: {accuracy:.2f}%")
    return optimizer

def evaluate_model(model, valid_loader, criterion, device):
    """
    Evaluates the model using validation data, returning loss and accuracy.
    """
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            total_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            correct_predictions += torch.sum(top_class == labels.view(*top_class.shape)).item()

    accuracy = 100 * correct_predictions / len(valid_loader.dataset)
    return total_loss, accuracy

def main():
    """
    Main entry point to load data, train the model, validate it, and save a checkpoint.
    """
    args = get_input_args_train()
    validate_train_args(args)

    # Load and transform datasets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(args.data_dir + '/train', transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize and train the model
    model = select_model_architecture(args.arch)
    model = configure_model_classifier(model, args.arch, args.hidden_units)
    optimizer = train_model(model, train_loader, None, args.learning_rate, args.epochs, args.gpu)

    # Save the trained model
    save_checkpoint(model, args.arch, args.save_dir, train_dataset, args.epochs, optimizer, model.classifier)

if __name__ == '__main__':
    main()

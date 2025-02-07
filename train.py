import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn, optim
from torchvision import datasets, transforms, models

def parse_arguments():
    """
    Parses command-line arguments for training the image classifier.
    """
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--arch', type=str, default="vgg16", help='Model architecture (default: vgg16)')
    parser.add_argument('--save_dir', type=str, default="./checkpoint.pth", help='Directory to save the model checkpoint')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=120, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    return parser.parse_args()

def apply_train_transforms(train_dir):
    """
    Applies transformations for training data.
    """
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return datasets.ImageFolder(train_dir, transform=train_transforms)

def apply_test_transforms(test_dir):
    """
    Applies transformations for testing/validation data.
    """
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return datasets.ImageFolder(test_dir, transform=test_transforms)

def create_dataloader(data, is_train=True):
    """
    Creates a data loader for the given dataset.
    """
    return torch.utils.data.DataLoader(data, batch_size=50, shuffle=is_train)

def get_device(use_gpu):
    """
    Determines the device to use (GPU or CPU).
    """
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        if use_gpu:
            print("GPU selected but not available. Falling back to CPU.")
        return torch.device("cpu")

def load_pretrained_model(architecture="vgg16"):
    """
    Loads a pre-trained model based on the specified architecture.
    """
    model = models.__dict__[architecture](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model

def build_classifier(input_size, hidden_units):
    """
    Builds a custom classifier for the model.
    """
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

def validate_model(model, dataloader, criterion, device):
    """
    Validates the model on a given dataset.
    """
    model.eval()
    loss, accuracy = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            predictions = torch.exp(outputs).max(dim=1)[1]
            accuracy += (predictions == labels).type(torch.FloatTensor).mean().item()
    return loss / len(dataloader), accuracy / len(dataloader)

def train_model(model, trainloader, validloader, device, criterion, optimizer, epochs):
    """
    Trains the model and evaluates it periodically.
    """
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        valid_loss, valid_accuracy = validate_model(model, validloader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs}... "
              f"Train Loss: {running_loss / len(trainloader):.4f}... "
              f"Validation Loss: {valid_loss:.4f}... "
              f"Validation Accuracy: {valid_accuracy * 100:.2f}%")

def save_checkpoint(model, save_path, train_data):
    """
    Saves the trained model checkpoint.
    """
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'architecture': model.name,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to {save_path}")

def main():
    """
    Main function to train the image classifier.
    """
    args = parse_arguments()

    train_data = apply_train_transforms('flowers/train')
    valid_data = apply_test_transforms('flowers/valid')
    trainloader = create_dataloader(train_data, is_train=True)
    validloader = create_dataloader(valid_data, is_train=False)

    model = load_pretrained_model(args.arch)
    model.classifier = build_classifier(25088, args.hidden_units)

    device = get_device(args.gpu)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train_model(model, trainloader, validloader, device, criterion, optimizer, args.epochs)

    save_checkpoint(model, args.save_dir, train_data)
    print("Training complete!")

if __name__ == '__main__':
    main()
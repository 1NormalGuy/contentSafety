import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from models.simple_cnn import SimpleCNN
from attack.pgd import PGDAttack

def evaluate_model(model, test_loader, device):
    """Evaluate model performance on clean test set"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def evaluate_pgd_attack(model, test_loader, device, eps=0.3, steps=40):
    """Evaluate model performance under PGD attack"""
    pgd_attack = PGDAttack(model, eps=eps, steps=steps)
    model.eval()
    correct = 0
    total = 0
    
    # Save some samples for visualization
    original_images = []
    adversarial_images = []
    original_labels = []
    adversarial_preds = []
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Cannot use with torch.no_grad() here
        adv_images = pgd_attack.attack(images, labels)
        
        # Predictions on adversarial samples can be done in no_grad mode
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if batch_idx == 0:
            original_images = images.detach().cpu()
            adversarial_images = adv_images.detach().cpu()
            original_labels = labels.detach().cpu()
            adversarial_preds = predicted.detach().cpu()
            
        if batch_idx >= 10:
            break

    visualize_results(original_images[:5], adversarial_images[:5], 
                     original_labels[:5], adversarial_preds[:5])
            
    adv_accuracy = 100 * correct / total
    return adv_accuracy

def visualize_results(original_images, adversarial_images, original_labels, adversarial_preds):
    """Visualize attack results"""
    os.makedirs('results', exist_ok=True)
    
    def denormalize(x):
        return x * 0.5 + 0.5
    
    plt.figure(figsize=(15, 10))
    
    for i in range(min(5, len(original_images))):
        # Original images
        plt.subplot(3, 5, i + 1)
        plt.imshow(denormalize(original_images[i][0]), cmap='gray')
        plt.title(f'Original: {original_labels[i].item()}')
        plt.axis('off')
        
        # Adversarial examples
        plt.subplot(3, 5, i + 6)
        plt.imshow(denormalize(adversarial_images[i][0]), cmap='gray')
        plt.title(f'Adversarial: {adversarial_preds[i].item()}')
        plt.axis('off')
        
        plt.subplot(3, 5, i + 11)
        perturbation = adversarial_images[i][0] - original_images[i][0]
        plt.imshow(perturbation, cmap='coolwarm', vmin=-0.3, vmax=0.3)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Perturbation')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/pgd_attack_results.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading MNIST test dataset...")
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Loading model...")
    model = SimpleCNN().to(device)
    
    try:
        model.load_state_dict(torch.load('mnist_cnn.pth'))
        print("Successfully loaded pre-trained model")
    except FileNotFoundError:
        print("Pre-trained model not found, training new model...")
        from train import train_model
        
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(5):
            model.train()
            running_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if batch_idx % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
            
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy:.2f}%')
        
        torch.save(model.state_dict(), 'mnist_cnn.pth')
        print("Model training completed and saved")

    print("\nEvaluating model performance on clean test set...")
    accuracy = evaluate_model(model, test_loader, device)
    print(f'Clean test set accuracy: {accuracy:.2f}%')

    print("\nEvaluating model under PGD attack...")
    for eps in [0.1, 0.2, 0.3]:
        for steps in [10, 20, 40]:
            print(f"\n- Parameters: eps={eps}, steps={steps}")
            adv_accuracy = evaluate_pgd_attack(model, test_loader, device, eps=eps, steps=steps)
            print(f'  Adversarial accuracy: {adv_accuracy:.2f}%')

if __name__ == "__main__":
    main()
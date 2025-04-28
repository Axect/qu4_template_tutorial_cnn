import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import beaupy
from rich.console import Console
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from util import (
    select_project,
    select_group,
    select_seed,
    select_device,
    load_model,
    load_data,
    load_study,
    load_best_model,
)


def test_model(model, dl_val, device):
    """Test a MNIST classification model and calculate metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dl_val:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = F.cross_entropy(y_pred, y)
            total_loss += loss.item()
            
            # Get predicted class
            _, predicted = torch.max(y_pred, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
            # Calculate accuracy
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dl_val)
    
    return avg_loss, accuracy, all_preds, all_targets


def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix for classification results"""
    plt.style.use(['science', 'nature'])
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=600, bbox_inches='tight')


def plot_examples(dl_val, model, device, num_examples=10):
    """Plot some example predictions"""
    model.eval()
    plt.style.use(['science', 'nature'])
    
    # Get batch of images
    for images, labels in dl_val:
        break
    
    images = images[:num_examples].to(device)
    labels = labels[:num_examples]
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Plot images with predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 7))
    axes = axes.flatten()
    
    for i in range(num_examples):
        axes[i].imshow(images[i].cpu().squeeze(), cmap='gray')
        correct = "(O)" if predicted[i].item() == labels[i].item() else "(X)"
        axes[i].set_title(f"Pred: {predicted[i].item()} {correct}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('examples.png', dpi=600, bbox_inches='tight')


def main():
    # Test run
    console = Console()
    console.print("[bold green]Analyzing the MNIST classification model...[/bold green]")
    console.print("Select a project to analyze:")
    project = select_project()
    console.print("Select a group to analyze:")
    group_name = select_group(project)
    console.print("Select a seed to analyze:")
    seed = select_seed(project, group_name)
    console.print("Select a device:")
    device = select_device()
    model, config = load_model(project, group_name, seed)
    model = model.to(device)

    # Load data
    train_dataset, val_dataset = load_data()
    dl_val = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Test the model
    val_loss, accuracy, all_preds, all_targets = test_model(model, dl_val, device)
    console.print(f"[bold]Validation Loss:[/bold] {val_loss:.4f}")
    console.print(f"[bold]Accuracy:[/bold] {accuracy:.2f}%")
    
    # Ask if user wants to see more detailed analysis
    if beaupy.confirm("Would you like to see a detailed classification report?"):
        console.print("\n[bold]Classification Report:[/bold]")
        report = classification_report(all_targets, all_preds, target_names=[str(i) for i in range(10)])
        console.print(report)
    
    if beaupy.confirm("Would you like to see the confusion matrix?"):
        plot_confusion_matrix(all_targets, all_preds, classes=[str(i) for i in range(10)])
    
    if beaupy.confirm("Would you like to see some example predictions?"):
        plot_examples(dl_val, model, device)


if __name__ == "__main__":
    console = Console()
    main()

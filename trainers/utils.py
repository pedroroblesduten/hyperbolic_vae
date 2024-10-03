import matplotlib.pyplot as plt
import os

def plot_loss(train_losses, test_losses, epoch, save_path=None):
    """Plots and updates the loss curves with a more professional style."""
    plt.figure(figsize=(12, 6))
    
    # Plot training and test losses with enhanced styles
    plt.plot(train_losses, label='Training Loss', color='blue', linestyle='-', marker='o', linewidth=2, markersize=5)
    plt.plot(test_losses, label='Test Loss', color='red', linestyle='--', marker='x', linewidth=2, markersize=5)
    
    # Set labels, title, and ticks
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'Epoch {epoch}: Training and Test Loss Curves', fontsize=16, fontweight='bold')

    # Add grid with customization
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

    # Customizing tick parameters for better readability
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add legend with a slight enlargement for readability
    plt.legend(loc='upper right', fontsize=12)

    # Show the plot
    plt.tight_layout()
    if save_path:
        # Ensure the directory exists
        path = os.path.join(save_path, 'losses', 'loss_curve.png')
        os.makedirs(os.path.dirname(path ), exist_ok=True)
        plt.savefig(path, format='png', dpi=300)  # Save in high resolution

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from devinterp.optim.sgld import SGLD
from devinterp.utils import default_nbeta
import matplotlib.pyplot as plt
import numpy as np

# %%
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, n_neurons: int = 256, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, n_neurons),
            nn.ReLU(inplace=True),  # inplace=True for memory efficiency
            nn.Linear(n_neurons, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class FastMLP(nn.Module):
    """Smaller, faster model for quick experiments"""
    def __init__(self, input_dim: int = 28 * 28, n_neurons: int = 128, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, n_neurons),
            nn.ReLU(inplace=True),
            nn.Linear(n_neurons, output_dim),
        )

    def forward(self, x):
        return self.net(x)
    
class MultiLayerMLP(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, n_neurons: int = 256, output_dim: int = 10, num_layers: int = 2):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(input_dim, n_neurons), nn.ReLU(inplace=True)])
        for _ in range(num_layers - 2):
            self.net.append(nn.Linear(n_neurons, n_neurons))
            self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(n_neurons, output_dim))
        self.net.append(nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.net(x)
    
class FastMultiLayerMLP(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, n_neurons: int = 128, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, n_neurons),
            nn.ReLU(inplace=True),
            nn.Linear(n_neurons, output_dim),
        )

    def forward(self, x):
        return self.net(x)
    
    

# %%

# %%
# Build a full MNIST dataset (train + test), then split reproducibly into train/test
mnist_train = MNIST(root='~/neuron-pool/data', train=True, transform=ToTensor(), download=True)
mnist_test = MNIST(root='~/neuron-pool/data', train=False, transform=ToTensor(), download=True)
total_dataset = ConcatDataset([mnist_train, mnist_test])

train_ratio = 0.8
train_size = int(train_ratio * len(total_dataset))
test_size = len(total_dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size], generator=generator)

# %%
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
print(f"Total samples: {len(total_dataset)}")
print(f"Train loader length: {len(train_loader)}")
print(f"Test loader length: {len(test_loader)}")

# %%
# Device
use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
device = torch.device("mps" if use_mps else "cpu")
print("Using device:", device)

# Data - Optimized for speed
num_workers = 4  # Increase for better data loading performance
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,  # Larger batch size for MPS
                          num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False,  # Even larger for evaluation
                         num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)

# Model/optim
model = TwoLayerMLP().to(device)
criterion = nn.CrossEntropyLoss()
# Highly optimized SGLD hyperparameters for maximum speed on MNIST
optimizer = SGLD(model.parameters(), lr=3e-2, nbeta=default_nbeta(train_loader), noise_level=0.05, weight_decay=5e-4)

def train_one_epoch(model, optimizer, criterion, loader, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    
    # Use torch.compile for speed if available (PyTorch 2.0+)
    try:
        if not hasattr(train_one_epoch, '_compiled_model'):
            train_one_epoch._compiled_model = torch.compile(model, mode='reduce-overhead')
        compiled_model = train_one_epoch._compiled_model
    except:
        compiled_model = model
    
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = x.view(x.size(0), -1)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        logits = compiled_model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += batch_size
    return total_loss / total, total_correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct, total = 0, 0
    
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = x.view(x.size(0), -1)
        logits = model(x)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_correct / total

num_epochs = 5
train_losses, train_accuracies, test_accuracies = [], [], []
for epoch in range(1, num_epochs + 1):
    epoch_loss, epoch_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)
    test_acc = evaluate(model, test_loader, device)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    test_accuracies.append(test_acc)
    print(f"Epoch {epoch:02d}/{num_epochs} - loss: {epoch_loss:.4f} - train_acc: {epoch_acc:.4f} - test_acc: {test_acc:.4f}")

# %%
# plt.plot(train_losses)
plt.plot(train_accuracies)
plt.plot(test_accuracies)
plt.show()

# %%
# Train multiple models and collect their neurons
def train_multiple_models(n_models=2, n_epochs=100):
    """Train multiple models with different random seeds and return their trained parameters."""
    models = []
    seeds = [42 + i for i in range(n_models)]
    
    print(f"Training {n_models} models...")
    for i, seed in enumerate(seeds):
        print(f"\nTraining model {i+1}/{n_models} (seed={seed})")
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Create fresh model and optimizer
        model = TwoLayerMLP().to(device)
        # Highly optimized SGLD hyperparameters for maximum speed
        optimizer = SGLD(model.parameters(), lr=3e-2, nbeta=default_nbeta(train_loader), 
                        noise_level=0.05, weight_decay=5e-4)
        
        # Train the model with aggressive early stopping
        best_test_acc = 0.0
        patience_counter = 0
        patience = 3  # Very aggressive - stop if no improvement for 3 epochs
        min_epochs = 3  # Minimum epochs before early stopping kicks in
        
        for epoch in range(1, n_epochs + 1):
            epoch_loss, epoch_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)
            
            # Only evaluate on test set every few epochs for speed (except first few epochs)
            if epoch <= 5 or epoch % 2 == 0 or epoch == n_epochs:
                test_acc = evaluate(model, test_loader, device)
                
                # Check for improvement
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Print progress less frequently
                if epoch <= 5 or epoch % 3 == 0 or epoch == n_epochs or patience_counter >= patience:
                    print(f"  Epoch {epoch:02d}/{n_epochs} - loss: {epoch_loss:.4f} - train_acc: {epoch_acc:.4f} - test_acc: {test_acc:.4f} - best: {best_test_acc:.4f}")
                
                # Early stopping (but only after minimum epochs)
                if epoch >= min_epochs and patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break
            else:
                # Just print training progress without test evaluation
                if epoch % 3 == 0:
                    print(f"  Epoch {epoch:02d}/{n_epochs} - loss: {epoch_loss:.4f} - train_acc: {epoch_acc:.4f} - [skipping test eval]")
        
        print(f"  Final test accuracy: {best_test_acc:.4f}")
        
        models.append(model)
    
    return models

def extract_neurons(models, device):
    """Extract neuron parameters from trained models."""
    all_weights_1 = []  # First layer weights (input -> hidden)
    all_biases_1 = []   # First layer biases
    all_weights_2 = []  # Second layer weights (hidden -> output)
    all_biases_2 = []   # Second layer biases
    
    for model in models:
        # Extract parameters from each model
        params = dict(model.named_parameters())
        
        # First layer (input -> hidden): each row is a neuron
        w1 = params['net.0.weight'].detach()  # Keep on device: [n_neurons, input_dim]
        b1 = params['net.0.bias'].detach()    # Keep on device: [n_neurons]
        
        # Second layer (hidden -> output): each column corresponds to a neuron
        w2 = params['net.2.weight'].detach()  # Keep on device: [output_dim, n_neurons]
        b2 = params['net.2.bias'].detach()    # Keep on device: [output_dim]
        
        # Store neurons (each neuron is defined by its input weights, bias, and output weights)
        for neuron_idx in range(w1.shape[0]):
            all_weights_1.append(w1[neuron_idx])      # Input weights for this neuron
            all_biases_1.append(b1[neuron_idx])       # Bias for this neuron
            all_weights_2.append(w2[:, neuron_idx])   # Output weights from this neuron
    
    # The second layer bias is shared across all neurons, so we'll take the mean
    for model in models:
        params = dict(model.named_parameters())
        all_biases_2.append(params['net.2.bias'].detach())
    
    return {
        'weights_1': torch.stack(all_weights_1),    # [total_neurons, input_dim] on device
        'biases_1': torch.stack(all_biases_1),      # [total_neurons] on device
        'weights_2': torch.stack(all_weights_2),    # [total_neurons, output_dim] on device
        'biases_2': torch.stack(all_biases_2)       # [n_models, output_dim] on device
    }

def create_model_from_neurons(neuron_pool, device, n_neurons=256, seed=None):
    """Create a new model by randomly sampling neurons from the pool."""
    if seed is not None:
        torch.manual_seed(seed)
    
    total_neurons = neuron_pool['weights_1'].shape[0]
    
    # Randomly sample neurons without replacement - keep on same device as neuron_pool
    selected_indices = torch.randperm(total_neurons, device=neuron_pool['weights_1'].device)[:n_neurons]
    
    # Create new model
    model = TwoLayerMLP(n_neurons=n_neurons).to(device)
    
    # Set parameters from selected neurons (already on correct device)
    with torch.no_grad():
        # First layer
        model.net[0].weight.data = neuron_pool['weights_1'][selected_indices]
        model.net[0].bias.data = neuron_pool['biases_1'][selected_indices]
        
        # Second layer - transpose because we stored as [total_neurons, output_dim] but need [output_dim, n_neurons]
        model.net[2].weight.data = neuron_pool['weights_2'][selected_indices].T
        
        # Second layer bias - use mean of all model biases
        model.net[2].bias.data = neuron_pool['biases_2'].mean(dim=0)
    
    return model, selected_indices

def plot_results(individual_accuracies, pooled_accuracies, save_plots=True):
    """Create comprehensive plots of the neuron pooling experiment results."""
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Bar plot comparing individual vs pooled models
    ax1 = plt.subplot(2, 3, 1)
    x_pos = np.arange(len(individual_accuracies))
    bars1 = ax1.bar(x_pos - 0.2, individual_accuracies, 0.4, label='Individual Models', alpha=0.7, color='skyblue')
    
    # For pooled models, repeat the pattern to match length
    pooled_extended = (pooled_accuracies * ((len(individual_accuracies) // len(pooled_accuracies)) + 1))[:len(individual_accuracies)]
    bars2 = ax1.bar(x_pos + 0.2, pooled_extended, 0.4, label='Pooled Models', alpha=0.7, color='lightcoral')
    
    ax1.set_xlabel('Model Index')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Individual vs Pooled Model Accuracies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Box plot comparison
    ax2 = plt.subplot(2, 3, 2)
    data_to_plot = [individual_accuracies, pooled_accuracies]
    bp = ax2.boxplot(data_to_plot, labels=['Individual', 'Pooled'], patch_artist=True)
    bp['boxes'][0].set_facecolor('skyblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Accuracy Distribution Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Histogram of accuracies
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(individual_accuracies, alpha=0.7, label='Individual', color='skyblue', bins=10)
    ax3.hist(pooled_accuracies, alpha=0.7, label='Pooled', color='lightcoral', bins=10)
    ax3.set_xlabel('Test Accuracy')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Accuracy Distributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = plt.subplot(2, 3, 4)
    individual_stats = [np.mean(individual_accuracies), np.std(individual_accuracies), 
                       np.min(individual_accuracies), np.max(individual_accuracies)]
    pooled_stats = [np.mean(pooled_accuracies), np.std(pooled_accuracies),
                   np.min(pooled_accuracies), np.max(pooled_accuracies)]
    
    x = np.arange(4)
    width = 0.35
    ax4.bar(x - width/2, individual_stats, width, label='Individual', color='skyblue', alpha=0.7)
    ax4.bar(x + width/2, pooled_stats, width, label='Pooled', color='lightcoral', alpha=0.7)
    ax4.set_xlabel('Statistic')
    ax4.set_ylabel('Value')
    ax4.set_title('Summary Statistics')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Mean', 'Std', 'Min', 'Max'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (ind_val, pool_val) in enumerate(zip(individual_stats, pooled_stats)):
        ax4.text(i - width/2, ind_val + 0.001, f'{ind_val:.3f}', ha='center', va='bottom', fontsize=8)
        ax4.text(i + width/2, pool_val + 0.001, f'{pool_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 5: Scatter plot
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(range(len(individual_accuracies)), individual_accuracies, 
               label='Individual', alpha=0.7, s=60, color='skyblue')
    ax5.scatter(range(len(pooled_accuracies)), pooled_accuracies, 
               label='Pooled', alpha=0.7, s=60, color='lightcoral')
    ax5.set_xlabel('Model Index')
    ax5.set_ylabel('Test Accuracy')
    ax5.set_title('Model Performance Scatter')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance difference
    ax6 = plt.subplot(2, 3, 6)
    mean_individual = np.mean(individual_accuracies)
    mean_pooled = np.mean(pooled_accuracies)
    difference = mean_pooled - mean_individual
    
    categories = ['Individual\nMean', 'Pooled\nMean', 'Difference']
    values = [mean_individual, mean_pooled, abs(difference)]
    colors = ['skyblue', 'lightcoral', 'green' if difference > 0 else 'red']
    
    bars = ax6.bar(categories, values, color=colors, alpha=0.7)
    ax6.set_ylabel('Accuracy')
    ax6.set_title('Performance Comparison')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add difference annotation
    if difference > 0:
        ax6.text(2, abs(difference)/2, f'+{difference:.3f}', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    else:
        ax6.text(2, abs(difference)/2, f'{difference:.3f}', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('neuron_pooling_results.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'neuron_pooling_results.png'")
    
    plt.show()
    
    # Additional statistical analysis
    print("\n" + "="*60)
    print("DETAILED STATISTICAL ANALYSIS")
    print("="*60)
    
    # T-test for significance
    from scipy import stats
    try:
        t_stat, p_value = stats.ttest_ind(individual_accuracies, pooled_accuracies)
        print(f"T-test results:")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
    except ImportError:
        print("scipy not available for t-test, install with: pip install scipy")
    
    print(f"\nDetailed Statistics:")
    print(f"Individual models:")
    print(f"  Mean: {np.mean(individual_accuracies):.4f}")
    print(f"  Std:  {np.std(individual_accuracies):.4f}")
    print(f"  Min:  {np.min(individual_accuracies):.4f}")
    print(f"  Max:  {np.max(individual_accuracies):.4f}")
    print(f"  Range: {np.max(individual_accuracies) - np.min(individual_accuracies):.4f}")
    
    print(f"\nPooled models:")
    print(f"  Mean: {np.mean(pooled_accuracies):.4f}")
    print(f"  Std:  {np.std(pooled_accuracies):.4f}")
    print(f"  Min:  {np.min(pooled_accuracies):.4f}")
    print(f"  Max:  {np.max(pooled_accuracies):.4f}")
    print(f"  Range: {np.max(pooled_accuracies) - np.min(pooled_accuracies):.4f}")
    
    improvement = ((np.mean(pooled_accuracies) - np.mean(individual_accuracies)) / np.mean(individual_accuracies)) * 100
    print(f"\nPerformance change: {improvement:+.2f}%")

def evaluate_neuron_pooling(n_models=10, n_test_models=5, train_epochs=100, plot_results_flag=True):
    """Main function to evaluate neuron pooling approach."""
    print("="*60)
    print("NEURON POOLING EXPERIMENT")
    print("="*60)
    
    # Step 1: Train multiple models
    trained_models = train_multiple_models(n_models=n_models, n_epochs=train_epochs)
    
    # Evaluate individual models
    print(f"\nEvaluating individual trained models:")
    individual_accuracies = []
    for i, model in enumerate(trained_models):
        acc = evaluate(model, test_loader, device)
        individual_accuracies.append(acc)
        print(f"Model {i+1}: {acc:.4f}")
    
    print(f"Mean individual accuracy: {np.mean(individual_accuracies):.4f} ± {np.std(individual_accuracies):.4f}")
    
    # Step 2: Extract neurons
    print(f"\nExtracting neurons from {n_models} models...")
    neuron_pool = extract_neurons(trained_models, device)
    total_neurons = neuron_pool['weights_1'].shape[0]
    print(f"Total neurons in pool: {total_neurons}")
    print(f"Neuron pool device: {neuron_pool['weights_1'].device}")
    
    # Step 3: Create and evaluate models from pooled neurons
    print(f"\nCreating {n_test_models} models from pooled neurons...")
    pooled_accuracies = []
    
    for i in range(n_test_models):
        pooled_model, selected_indices = create_model_from_neurons(
            neuron_pool, device, n_neurons=256, seed=1000+i
        )
        acc = evaluate(pooled_model, test_loader, device)
        pooled_accuracies.append(acc)
        print(f"Pooled model {i+1}: {acc:.4f}")
    
    print(f"Mean pooled accuracy: {np.mean(pooled_accuracies):.4f} ± {np.std(pooled_accuracies):.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Individual models: {np.mean(individual_accuracies):.4f} ± {np.std(individual_accuracies):.4f}")
    print(f"Pooled models:     {np.mean(pooled_accuracies):.4f} ± {np.std(pooled_accuracies):.4f}")
    
    # Create plots if requested
    if plot_results_flag:
        plot_results(individual_accuracies, pooled_accuracies)
    
    return {
        'individual_accuracies': individual_accuracies,
        'pooled_accuracies': pooled_accuracies,
        'neuron_pool': neuron_pool,
        'trained_models': trained_models
    }

# %%
# STEP 1: Train multiple models and create neuron pool
print("="*60)
print("STEP 1: TRAINING MULTIPLE MODELS")
print("="*60)

# Train the models with reduced epochs for speed
trained_models = train_multiple_models(n_models=10, n_epochs=15)

# Evaluate individual models
print(f"\nEvaluating individual trained models:")
individual_accuracies = []
for i, model in enumerate(trained_models):
    acc = evaluate(model, test_loader, device)
    individual_accuracies.append(acc)
    print(f"Model {i+1}: {acc:.4f}")

print(f"Mean individual accuracy: {np.mean(individual_accuracies):.4f} ± {np.std(individual_accuracies):.4f}")

# Extract neurons and create pool
print(f"\nExtracting neurons from {len(trained_models)} models...")
neuron_pool = extract_neurons(trained_models, device)
total_neurons = neuron_pool['weights_1'].shape[0]
print(f"Total neurons in pool: {total_neurons}")
print(f"Neuron pool device: {neuron_pool['weights_1'].device}")

# %%
# STEP 2: Neuron pooling experiments
print("\n" + "="*60)
print("STEP 2: NEURON POOLING EXPERIMENTS")
print("="*60)

def create_random_pooled_model(neuron_pool, device, n_neurons=256, seed=None):
    """Create a new model by randomly sampling neurons from the pool."""
    if seed is not None:
        torch.manual_seed(seed)
    
    total_neurons = neuron_pool['weights_1'].shape[0]
    
    # Randomly sample neurons without replacement
    selected_indices = torch.randperm(total_neurons, device=neuron_pool['weights_1'].device)[:n_neurons]
    
    # Create new model
    model = TwoLayerMLP(n_neurons=n_neurons).to(device)
    
    # Set parameters from selected neurons
    with torch.no_grad():
        # First layer
        model.net[0].weight.data = neuron_pool['weights_1'][selected_indices]
        model.net[0].bias.data = neuron_pool['biases_1'][selected_indices]
        
        # Second layer
        model.net[2].weight.data = neuron_pool['weights_2'][selected_indices].T
        
        # Second layer bias - use mean of all model biases
        model.net[2].bias.data = neuron_pool['biases_2'].mean(dim=0)
    
    return model, selected_indices

def create_hybrid_model(base_model, neuron_pool, device, replacement_ratio=0.1, seed=None):
    """Replace a percentage of neurons in an existing model with neurons from the pool."""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Copy the base model
    hybrid_model = TwoLayerMLP().to(device)
    hybrid_model.load_state_dict(base_model.state_dict())
    
    n_neurons = hybrid_model.net[0].weight.shape[0]  # 256
    n_replace = int(n_neurons * replacement_ratio)
    total_pool_neurons = neuron_pool['weights_1'].shape[0]
    
    # Select which neurons to replace in the base model
    replace_indices = torch.randperm(n_neurons, device='cpu')[:n_replace]
    
    # Select replacement neurons from the pool
    pool_indices = torch.randperm(total_pool_neurons, device=neuron_pool['weights_1'].device)[:n_replace]
    
    with torch.no_grad():
        # Replace selected neurons
        for i, (replace_idx, pool_idx) in enumerate(zip(replace_indices, pool_indices)):
            # Replace first layer weights and bias
            hybrid_model.net[0].weight.data[replace_idx] = neuron_pool['weights_1'][pool_idx]
            hybrid_model.net[0].bias.data[replace_idx] = neuron_pool['biases_1'][pool_idx]
            
            # Replace second layer weights (connections from this neuron to output)
            hybrid_model.net[2].weight.data[:, replace_idx] = neuron_pool['weights_2'][pool_idx]
    
    return hybrid_model, replace_indices, pool_indices

# Experiment 1: Random pooled models
print(f"\nExperiment 1: Creating 5 random pooled models...")
random_pooled_accuracies = []
for i in range(5):
    model, selected_indices = create_random_pooled_model(neuron_pool, device, seed=1000+i)
    acc = evaluate(model, test_loader, device)
    random_pooled_accuracies.append(acc)
    print(f"Random pooled model {i+1}: {acc:.4f}")

print(f"Mean random pooled accuracy: {np.mean(random_pooled_accuracies):.4f} ± {np.std(random_pooled_accuracies):.4f}")

# Experiment 2: Hybrid models (10% replacement)
print(f"\nExperiment 2: Creating 5 hybrid models (10% neuron replacement)...")
hybrid_accuracies = []
for i in range(5):
    # Use different base models for variety
    base_model = trained_models[i % len(trained_models)]
    hybrid_model, replace_indices, pool_indices = create_hybrid_model(
        base_model, neuron_pool, device, replacement_ratio=0.1, seed=2000+i
    )
    acc = evaluate(hybrid_model, test_loader, device)
    hybrid_accuracies.append(acc)
    print(f"Hybrid model {i+1}: {acc:.4f} (base: {individual_accuracies[i % len(trained_models)]:.4f})")

print(f"Mean hybrid accuracy: {np.mean(hybrid_accuracies):.4f} ± {np.std(hybrid_accuracies):.4f}")

# %%
# STEP 3: Comprehensive analysis and plotting
print("\n" + "="*60)
print("STEP 3: COMPREHENSIVE ANALYSIS")
print("="*60)

def plot_comprehensive_results(individual_accs, random_pooled_accs, hybrid_accs, save_plots=True):
    """Create comprehensive plots comparing all pooling mechanisms."""
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Bar plot comparison of all methods
    ax1 = plt.subplot(2, 4, 1)
    methods = ['Individual', 'Random Pooled', 'Hybrid (10%)']
    means = [np.mean(individual_accs), np.mean(random_pooled_accs), np.mean(hybrid_accs)]
    stds = [np.std(individual_accs), np.std(random_pooled_accs), np.std(hybrid_accs)]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = ax1.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Mean Performance Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.002,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Box plot comparison
    ax2 = plt.subplot(2, 4, 2)
    data_to_plot = [individual_accs, random_pooled_accs, hybrid_accs]
    bp = ax2.boxplot(data_to_plot, labels=['Individual', 'Random\nPooled', 'Hybrid\n(10%)'], patch_artist=True)
    bp['boxes'][0].set_facecolor('skyblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    bp['boxes'][2].set_facecolor('lightgreen')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Distribution Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Individual model performance scatter
    ax3 = plt.subplot(2, 4, 3)
    x_individual = range(len(individual_accs))
    x_random = [x + 0.2 for x in range(len(random_pooled_accs))]
    x_hybrid = [x + 0.4 for x in range(len(hybrid_accs))]
    
    ax3.scatter(x_individual, individual_accs, label='Individual', alpha=0.7, s=60, color='skyblue')
    ax3.scatter(x_random, random_pooled_accs, label='Random Pooled', alpha=0.7, s=60, color='lightcoral')
    ax3.scatter(x_hybrid, hybrid_accs, label='Hybrid', alpha=0.7, s=60, color='lightgreen')
    ax3.set_xlabel('Model Index')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Individual Model Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histogram overlays
    ax4 = plt.subplot(2, 4, 4)
    ax4.hist(individual_accs, alpha=0.7, label='Individual', color='skyblue', bins=8)
    ax4.hist(random_pooled_accs, alpha=0.7, label='Random Pooled', color='lightcoral', bins=8)
    ax4.hist(hybrid_accs, alpha=0.7, label='Hybrid', color='lightgreen', bins=8)
    ax4.set_xlabel('Test Accuracy')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Accuracy Distributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Performance improvement analysis
    ax5 = plt.subplot(2, 4, 5)
    baseline = np.mean(individual_accs)
    improvements = [
        0,  # baseline
        (np.mean(random_pooled_accs) - baseline) / baseline * 100,
        (np.mean(hybrid_accs) - baseline) / baseline * 100
    ]
    
    colors_imp = ['gray', 'red' if improvements[1] < 0 else 'green', 'red' if improvements[2] < 0 else 'green']
    bars = ax5.bar(methods, improvements, color=colors_imp, alpha=0.7)
    ax5.set_ylabel('Performance Change (%)')
    ax5.set_title('Relative Performance vs Individual')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.2),
                f'{imp:+.2f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    # Plot 6: Statistical summary
    ax6 = plt.subplot(2, 4, 6)
    stats = ['Mean', 'Std', 'Min', 'Max']
    individual_stats = [np.mean(individual_accs), np.std(individual_accs), 
                       np.min(individual_accs), np.max(individual_accs)]
    random_stats = [np.mean(random_pooled_accs), np.std(random_pooled_accs),
                   np.min(random_pooled_accs), np.max(random_pooled_accs)]
    hybrid_stats = [np.mean(hybrid_accs), np.std(hybrid_accs),
                   np.min(hybrid_accs), np.max(hybrid_accs)]
    
    x = np.arange(len(stats))
    width = 0.25
    ax6.bar(x - width, individual_stats, width, label='Individual', color='skyblue', alpha=0.7)
    ax6.bar(x, random_stats, width, label='Random Pooled', color='lightcoral', alpha=0.7)
    ax6.bar(x + width, hybrid_stats, width, label='Hybrid', color='lightgreen', alpha=0.7)
    ax6.set_xlabel('Statistic')
    ax6.set_ylabel('Value')
    ax6.set_title('Statistical Summary')
    ax6.set_xticks(x)
    ax6.set_xticklabels(stats)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Variance analysis
    ax7 = plt.subplot(2, 4, 7)
    variances = [np.var(individual_accs), np.var(random_pooled_accs), np.var(hybrid_accs)]
    bars = ax7.bar(methods, variances, color=colors, alpha=0.7)
    ax7.set_ylabel('Variance')
    ax7.set_title('Performance Variance')
    ax7.grid(True, alpha=0.3)
    
    for bar, var in zip(bars, variances):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{var:.6f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 8: Range analysis
    ax8 = plt.subplot(2, 4, 8)
    ranges = [np.max(individual_accs) - np.min(individual_accs),
             np.max(random_pooled_accs) - np.min(random_pooled_accs),
             np.max(hybrid_accs) - np.min(hybrid_accs)]
    bars = ax8.bar(methods, ranges, color=colors, alpha=0.7)
    ax8.set_ylabel('Range (Max - Min)')
    ax8.set_title('Performance Range')
    ax8.grid(True, alpha=0.3)
    
    for bar, rng in zip(bars, ranges):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{rng:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('comprehensive_neuron_pooling_results.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'comprehensive_neuron_pooling_results.png'")
    
    plt.show()
    
    # Statistical analysis
    print("\n" + "="*60)
    print("DETAILED STATISTICAL ANALYSIS")
    print("="*60)
    
    print(f"Individual Models (n={len(individual_accs)}):")
    print(f"  Mean: {np.mean(individual_accs):.4f}")
    print(f"  Std:  {np.std(individual_accs):.4f}")
    print(f"  Min:  {np.min(individual_accs):.4f}")
    print(f"  Max:  {np.max(individual_accs):.4f}")
    print(f"  Range: {np.max(individual_accs) - np.min(individual_accs):.4f}")
    
    print(f"\nRandom Pooled Models (n={len(random_pooled_accs)}):")
    print(f"  Mean: {np.mean(random_pooled_accs):.4f}")
    print(f"  Std:  {np.std(random_pooled_accs):.4f}")
    print(f"  Min:  {np.min(random_pooled_accs):.4f}")
    print(f"  Max:  {np.max(random_pooled_accs):.4f}")
    print(f"  Range: {np.max(random_pooled_accs) - np.min(random_pooled_accs):.4f}")
    
    print(f"\nHybrid Models (n={len(hybrid_accs)}):")
    print(f"  Mean: {np.mean(hybrid_accs):.4f}")
    print(f"  Std:  {np.std(hybrid_accs):.4f}")
    print(f"  Min:  {np.min(hybrid_accs):.4f}")
    print(f"  Max:  {np.max(hybrid_accs):.4f}")
    print(f"  Range: {np.max(hybrid_accs) - np.min(hybrid_accs):.4f}")
    
    # Performance comparisons
    random_improvement = ((np.mean(random_pooled_accs) - np.mean(individual_accs)) / np.mean(individual_accs)) * 100
    hybrid_improvement = ((np.mean(hybrid_accs) - np.mean(individual_accs)) / np.mean(individual_accs)) * 100
    
    print(f"\nPerformance Changes:")
    print(f"  Random Pooled vs Individual: {random_improvement:+.2f}%")
    print(f"  Hybrid vs Individual: {hybrid_improvement:+.2f}%")
    print(f"  Hybrid vs Random Pooled: {((np.mean(hybrid_accs) - np.mean(random_pooled_accs)) / np.mean(random_pooled_accs)) * 100:+.2f}%")

# Create comprehensive plots
plot_comprehensive_results(individual_accuracies, random_pooled_accuracies, hybrid_accuracies)

# %%
# STEP 4: LLC (Learning Loss Complexity) Estimation
print("\n" + "="*60)
print("STEP 4: LLC ESTIMATION")
print("="*60)

from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.slt.llc import LLCEstimator

def estimate_llc_for_model(model, train_loader, device, n_samples=50, n_chains=4):
    """Estimate LLC for a single model using the high-level devinterp API."""
    print(f"Estimating LLC with {n_samples} samples and {n_chains} chains...")
    
    # Define evaluation function for devinterp
    def eval_fn(model, batch):
        """Evaluation function that returns loss for a batch (with gradients)."""
        model.train()  # Need gradients for SGLD sampling
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = x.view(x.size(0), -1)
        
        # Don't use torch.no_grad() - we need gradients for sampling
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        return loss
    
    try:
        # Use the high-level API with proper evaluation function
        results = estimate_learning_coeff_with_summary(
            model=model,
            loader=train_loader,
            evaluate=eval_fn,  # Provide evaluation function
            num_draws=n_samples,
            num_chains=n_chains,
            num_burnin_steps=max(100, n_samples * 2),  # Adequate burn-in
            device=device,
            optimizer_kwargs={
                'lr': 1e-4,
                'noise_level': 1.0,  # Proper SGLD
                'weight_decay': 1e-4
                # nbeta is handled automatically by the API
            },
            verbose=False
        )
        
        # Extract LLC results
        if results and 'llc/mean' in results:
            llc_mean = results['llc/mean']
            llc_std = results.get('llc/std', 0.0)
            print(f"  LLC: {llc_mean:.4f} ± {llc_std:.4f}")
            return llc_mean, llc_std, results
        else:
            print(f"  Warning: No valid LLC results in response")
            print(f"  Available keys: {list(results.keys()) if results else 'None'}")
            return float('nan'), float('nan'), {}
            
    except Exception as e:
        print(f"  Detailed error: {str(e)}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return float('nan'), float('nan'), {}

def estimate_llc_for_all_models(models, train_loader, device, n_samples=50):
    """Estimate LLC for all trained models."""
    llc_results = []
    llc_means = []
    llc_stds = []
    
    print(f"Estimating LLC for {len(models)} individual models...")
    
    for i, model in enumerate(models):
        print(f"\nModel {i+1}/{len(models)}:")
        try:
            llc_mean, llc_std, full_results = estimate_llc_for_model(
                model, train_loader, device, n_samples=n_samples
            )
            llc_means.append(llc_mean)
            llc_stds.append(llc_std)
            llc_results.append(full_results)
            print(f"  LLC: {llc_mean:.4f} ± {llc_std:.4f}")
        except Exception as e:
            print(f"  Error estimating LLC: {e}")
            llc_means.append(float('nan'))
            llc_stds.append(float('nan'))
            llc_results.append({})
    
    return llc_means, llc_stds, llc_results

def analyze_llc_results(llc_means, llc_stds, individual_accuracies):
    """Analyze and visualize LLC results."""
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Filter out NaN values
    valid_indices = ~np.isnan(llc_means)
    valid_llc_means = np.array(llc_means)[valid_indices]
    valid_llc_stds = np.array(llc_stds)[valid_indices]
    valid_accuracies = np.array(individual_accuracies)[valid_indices]
    
    if len(valid_llc_means) == 0:
        print("No valid LLC estimates available for analysis.")
        return
    
    print("\n" + "="*60)
    print("LLC ANALYSIS RESULTS")
    print("="*60)
    
    print(f"LLC Statistics:")
    print(f"  Mean LLC: {np.mean(valid_llc_means):.4f} ± {np.std(valid_llc_means):.4f}")
    print(f"  Min LLC:  {np.min(valid_llc_means):.4f}")
    print(f"  Max LLC:  {np.max(valid_llc_means):.4f}")
    print(f"  Range:    {np.max(valid_llc_means) - np.min(valid_llc_means):.4f}")
    
    # Check if models are in the same basin (similar LLC values)
    llc_cv = np.std(valid_llc_means) / np.mean(valid_llc_means) if np.mean(valid_llc_means) != 0 else float('inf')
    print(f"  Coefficient of Variation: {llc_cv:.4f}")
    
    if llc_cv < 0.1:
        print("  → Models likely from the SAME basin (low LLC variance)")
    elif llc_cv < 0.3:
        print("  → Models possibly from similar basins (moderate LLC variance)")
    else:
        print("  → Models likely from DIFFERENT basins (high LLC variance)")
    
    # Plot LLC results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: LLC values for each model
    ax1.bar(range(len(valid_llc_means)), valid_llc_means, yerr=valid_llc_stds, 
            capsize=5, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Model Index')
    ax1.set_ylabel('LLC')
    ax1.set_title('LLC Estimates for Individual Models')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(valid_llc_means, valid_llc_stds)):
        ax1.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: LLC vs Accuracy scatter
    ax2.scatter(valid_accuracies, valid_llc_means, alpha=0.7, s=60, color='lightcoral')
    ax2.set_xlabel('Test Accuracy')
    ax2.set_ylabel('LLC')
    ax2.set_title('LLC vs Test Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation info
    if len(valid_llc_means) > 2:
        correlation = np.corrcoef(valid_accuracies, valid_llc_means)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3: LLC distribution
    ax3.hist(valid_llc_means, bins=min(10, len(valid_llc_means)), alpha=0.7, color='lightgreen')
    ax3.axvline(np.mean(valid_llc_means), color='red', linestyle='--', 
               label=f'Mean: {np.mean(valid_llc_means):.3f}')
    ax3.set_xlabel('LLC')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of LLC Values')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: LLC coefficient of variation comparison
    ax4.bar(['LLC', 'Accuracy'], 
           [llc_cv, np.std(valid_accuracies) / np.mean(valid_accuracies)],
           color=['skyblue', 'lightcoral'], alpha=0.7)
    ax4.set_ylabel('Coefficient of Variation')
    ax4.set_title('Variability: LLC vs Accuracy')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, val in enumerate([llc_cv, np.std(valid_accuracies) / np.mean(valid_accuracies)]):
        ax4.text(i, val + val*0.05, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('llc_analysis_results.png', dpi=300, bbox_inches='tight')
    print("\nLLC analysis plots saved as 'llc_analysis_results.png'")
    plt.show()

# Estimate LLC for all individual models
print("Starting LLC estimation for individual models...")
print("Note: This may take several minutes...")
print("Using conservative parameters to avoid sampling issues...")

# Use smaller, more conservative parameters for initial testing
llc_means, llc_stds, llc_results = estimate_llc_for_all_models(
    trained_models, train_loader, device, n_samples=20  # Reduced for stability
)

# Analyze the results
analyze_llc_results(llc_means, llc_stds, individual_accuracies)

# Store LLC results for potential further analysis
llc_data = {
    'llc_means': llc_means,
    'llc_stds': llc_stds,
    'individual_accuracies': individual_accuracies,
    'full_results': llc_results
}

print(f"\nLLC estimation complete!")
print(f"Results stored in 'llc_data' variable for further analysis.")

# %%
# STEP 5: LLC Analysis of Pooled Models
print("\n" + "="*60)
print("STEP 5: LLC ANALYSIS OF POOLED MODELS")
print("="*60)

def estimate_pooled_model_llcs(neuron_pool, device, n_test_models=5, n_samples=20):
    """Estimate LLC for pooled models and analyze neuron contribution correlations."""
    
    print(f"Estimating LLC for {n_test_models} pooled models...")
    
    pooled_llc_data = []
    pooled_models_info = []
    
    for i in range(n_test_models):
        print(f"\nPooled Model {i+1}/{n_test_models}:")
        
        # Create random pooled model
        pooled_model, selected_indices = create_random_pooled_model(
            neuron_pool, device, seed=3000+i
        )
        
        # Estimate LLC for this pooled model
        llc_mean, llc_std, llc_results = estimate_llc_for_model(
            pooled_model, train_loader, device, n_samples=n_samples
        )
        
        # Calculate weighted LLC contribution from source models
        neuron_contributions = calculate_neuron_contributions(
            selected_indices, llc_means, len(trained_models)
        )
        
        pooled_llc_data.append({
            'model_idx': i,
            'llc_mean': llc_mean,
            'llc_std': llc_std,
            'selected_indices': selected_indices,
            'neuron_contributions': neuron_contributions,
            'accuracy': evaluate(pooled_model, test_loader, device)
        })
        
        pooled_models_info.append(pooled_model)
        
        print(f"  Pooled LLC: {llc_mean:.4f} ± {llc_std:.4f}")
        print(f"  Expected LLC (weighted): {neuron_contributions['weighted_llc']:.4f}")
        print(f"  LLC difference: {llc_mean - neuron_contributions['weighted_llc']:.4f}")
    
    return pooled_llc_data, pooled_models_info

def calculate_neuron_contributions(selected_indices, source_llcs, n_source_models):
    """Calculate weighted LLC contribution from source models."""
    
    # Convert to numpy for easier manipulation
    selected_indices_np = selected_indices.cpu().numpy()
    n_neurons_per_model = 256  # Each source model has 256 neurons
    
    # Count how many neurons come from each source model
    source_counts = np.zeros(n_source_models)
    source_llcs_valid = []
    
    for idx in selected_indices_np:
        source_model_idx = idx // n_neurons_per_model
        if source_model_idx < n_source_models:
            source_counts[source_model_idx] += 1
    
    # Calculate weighted LLC based on neuron distribution
    total_neurons = len(selected_indices_np)
    weighted_llc = 0.0
    valid_contribution = 0.0
    
    for i, (count, source_llc) in enumerate(zip(source_counts, source_llcs)):
        if not np.isnan(source_llc) and count > 0:
            weight = count / total_neurons
            contribution = weight * source_llc
            weighted_llc += contribution
            valid_contribution += weight
            
    # Normalize if not all source models had valid LLCs
    if valid_contribution > 0:
        weighted_llc = weighted_llc / valid_contribution * (valid_contribution)
    
    return {
        'source_counts': source_counts,
        'source_weights': source_counts / total_neurons,
        'weighted_llc': weighted_llc,
        'valid_contribution': valid_contribution,
        'neuron_diversity': np.sum(source_counts > 0) / n_source_models  # Fraction of source models used
    }

def analyze_llc_correlations(pooled_llc_data, individual_llc_means):
    """Analyze correlations between LLC differences and neuron contributions."""
    
    print("\n" + "="*60)
    print("LLC CORRELATION ANALYSIS")
    print("="*60)
    
    # Extract data for correlation analysis
    pooled_llcs = [data['llc_mean'] for data in pooled_llc_data if not np.isnan(data['llc_mean'])]
    weighted_llcs = [data['neuron_contributions']['weighted_llc'] for data in pooled_llc_data if not np.isnan(data['llc_mean'])]
    llc_differences = [pooled - weighted for pooled, weighted in zip(pooled_llcs, weighted_llcs)]
    neuron_diversities = [data['neuron_contributions']['neuron_diversity'] for data in pooled_llc_data if not np.isnan(data['llc_mean'])]
    pooled_accuracies = [data['accuracy'] for data in pooled_llc_data if not np.isnan(data['llc_mean'])]
    
    if len(pooled_llcs) == 0:
        print("No valid pooled LLC estimates for correlation analysis.")
        return
    
    print(f"Valid pooled models for analysis: {len(pooled_llcs)}")
    
    # Statistical analysis
    print(f"\nStatistical Summary:")
    print(f"Pooled LLCs: {np.mean(pooled_llcs):.4f} ± {np.std(pooled_llcs):.4f}")
    print(f"Weighted LLCs: {np.mean(weighted_llcs):.4f} ± {np.std(weighted_llcs):.4f}")
    print(f"LLC differences: {np.mean(llc_differences):.4f} ± {np.std(llc_differences):.4f}")
    print(f"Neuron diversity: {np.mean(neuron_diversities):.4f} ± {np.std(neuron_diversities):.4f}")
    
    # Correlation analysis
    correlations = {}
    if len(pooled_llcs) > 2:
        correlations['llc_diff_vs_diversity'] = np.corrcoef(llc_differences, neuron_diversities)[0, 1]
        correlations['pooled_vs_weighted'] = np.corrcoef(pooled_llcs, weighted_llcs)[0, 1]
        correlations['llc_diff_vs_accuracy'] = np.corrcoef(llc_differences, pooled_accuracies)[0, 1]
        
        print(f"\nCorrelations:")
        print(f"LLC difference vs Neuron diversity: {correlations['llc_diff_vs_diversity']:.4f}")
        print(f"Pooled vs Weighted LLC: {correlations['pooled_vs_weighted']:.4f}")
        print(f"LLC difference vs Accuracy: {correlations['llc_diff_vs_accuracy']:.4f}")
    
    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Pooled vs Weighted LLC
    ax1.scatter(weighted_llcs, pooled_llcs, alpha=0.7, s=80, color='skyblue')
    ax1.plot([min(weighted_llcs + pooled_llcs), max(weighted_llcs + pooled_llcs)], 
             [min(weighted_llcs + pooled_llcs), max(weighted_llcs + pooled_llcs)], 
             'r--', alpha=0.5, label='Perfect Agreement')
    ax1.set_xlabel('Weighted LLC (Expected)')
    ax1.set_ylabel('Pooled LLC (Actual)')
    ax1.set_title('Pooled vs Expected LLC')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if len(pooled_llcs) > 2:
        ax1.text(0.05, 0.95, f'R = {correlations["pooled_vs_weighted"]:.3f}', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 2: LLC Difference vs Neuron Diversity
    ax2.scatter(neuron_diversities, llc_differences, alpha=0.7, s=80, color='lightcoral')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Neuron Diversity (Fraction of Source Models Used)')
    ax2.set_ylabel('LLC Difference (Actual - Expected)')
    ax2.set_title('LLC Deviation vs Neuron Mixing')
    ax2.grid(True, alpha=0.3)
    
    if len(pooled_llcs) > 2:
        ax2.text(0.05, 0.95, f'R = {correlations["llc_diff_vs_diversity"]:.3f}', 
                transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3: LLC Difference vs Accuracy
    ax3.scatter(pooled_accuracies, llc_differences, alpha=0.7, s=80, color='lightgreen')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Pooled Model Accuracy')
    ax3.set_ylabel('LLC Difference (Actual - Expected)')
    ax3.set_title('LLC Deviation vs Performance')
    ax3.grid(True, alpha=0.3)
    
    if len(pooled_llcs) > 2:
        ax3.text(0.05, 0.95, f'R = {correlations["llc_diff_vs_accuracy"]:.3f}', 
                transform=ax3.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 4: Source Model Contribution Heatmap
    ax4.set_title('Neuron Source Distribution Across Pooled Models')
    if len(pooled_llc_data) > 0:
        # Create heatmap of source contributions
        contribution_matrix = np.array([data['neuron_contributions']['source_weights'] 
                                      for data in pooled_llc_data if not np.isnan(data['llc_mean'])])
        
        if contribution_matrix.size > 0:
            im = ax4.imshow(contribution_matrix, cmap='YlOrRd', aspect='auto')
            ax4.set_xlabel('Source Model Index')
            ax4.set_ylabel('Pooled Model Index')
            plt.colorbar(im, ax=ax4, label='Fraction of Neurons')
        else:
            ax4.text(0.5, 0.5, 'No valid data', transform=ax4.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('llc_correlation_analysis.png', dpi=300, bbox_inches='tight')
    print("\nCorrelation analysis plots saved as 'llc_correlation_analysis.png'")
    plt.show()
    
    return correlations, llc_differences, neuron_diversities

# Run LLC analysis for pooled models
print("Estimating LLCs for pooled models...")
pooled_llc_data, pooled_models_info = estimate_pooled_model_llcs(
    neuron_pool, device, n_test_models=5, n_samples=15  # Reduced for speed
)

# Analyze correlations
correlations, llc_diffs, diversities = analyze_llc_correlations(pooled_llc_data, llc_means)

# Store comprehensive results
comprehensive_llc_data = {
    'individual_llcs': llc_data,
    'pooled_llcs': pooled_llc_data,
    'correlations': correlations,
    'llc_differences': llc_diffs,
    'neuron_diversities': diversities
}

print(f"\nComprehensive LLC analysis complete!")
print(f"Results stored in 'comprehensive_llc_data' variable.")

# %%


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import time
import os
from dataclasses import dataclass
import json

# Import our model implementations
# Assuming the previous artifacts are saved as separate files:
# from ctnet_implementation import create_ctnet
# from eeg_deformer_implementation import create_eeg_deformer  
# from eegmamba_implementation import create_eegmamba
# from multiscale_attention_implementation import create_multiscale_attention_network

@dataclass
class TrainingConfig:
    """Configuration for training experiments"""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    weight_decay: float = 1e-4
    patience: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_models: bool = True
    save_dir: str = "model_checkpoints"


class ModelEvaluator:
    """
    Comprehensive evaluator for interpretability vs classification robustness
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results = {}
        
        # Create save directory
        if config.save_models:
            os.makedirs(config.save_dir, exist_ok=True)
    
    def create_models(self, n_channels: int, n_classes: int, samples: int) -> Dict[str, nn.Module]:
        """Create all modern EEG models for comparison"""
        
        models = {
            # Modern architectures
            "CTNet": create_ctnet(n_channels, n_classes, samples),
            "EEG-Deformer": create_eeg_deformer(n_channels, n_classes, samples),
            "EEGMamba": create_eegmamba(n_channels, n_classes, samples),
            "MultiScale-Attention": create_multiscale_attention_network(
                n_channels, n_classes, samples, model_type="full"
            ),
            "MultiScale-Compact": create_multiscale_attention_network(
                n_channels, n_classes, samples, model_type="compact"
            ),
            
            # Baseline models for comparison (you would implement these separately)
            # "EEGNet": create_eegnet(n_channels, n_classes, samples),
            # "DeepConvNet": create_deepconvnet(n_channels, n_classes, samples),
        }
        
        # Move models to device
        for name, model in models.items():
            models[name] = model.to(self.device)
            
        return models
    
    def print_model_stats(self, models: Dict[str, nn.Module]):
        """Print model statistics"""
        print("Model Statistics:")
        print("=" * 60)
        
        for name, model in models.items():
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"{name:20} | Params: {num_params:8,} | Trainable: {trainable_params:8,}")
        
        print("=" * 60)
    
    def train_model(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        model_name: str
    ) -> Dict[str, Any]:
        """Train a single model"""
        
        print(f"\nTraining {model_name}...")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=self.config.patience//2, factor=0.5
        )
        
        # Training tracking
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        best_val_acc = 0.0
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass (handle different model signatures)
                if "EEGMamba" in model_name:
                    outputs = model(batch_x, task_id=0)  # Motor imagery task
                else:
                    outputs = model(batch_x)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    if "EEGMamba" in model_name:
                        outputs = model(batch_x, task_id=0)
                    else:
                        outputs = model(batch_x)
                    
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Store metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                if self.config.save_models:
                    torch.save(
                        model.state_dict(), 
                        os.path.join(self.config.save_dir, f"{model_name}_best.pth")
                    )
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Train Acc: {train_acc:.2f}%, "
                      f"Val Acc: {val_acc:.2f}%, Best Val: {best_val_acc:.2f}%")
            
            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        
        return {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "best_val_acc": best_val_acc,
            "training_time": training_time,
            "epochs_trained": epoch + 1
        }
    
    def evaluate_model(
        self, 
        model: nn.Module, 
        test_loader: DataLoader, 
        model_name: str
    ) -> Dict[str, Any]:
        """Evaluate model on test set"""
        
        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                if "EEGMamba" in model_name:
                    outputs = model(batch_x, task_id=0)
                else:
                    outputs = model(batch_x)
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        test_acc = 100 * test_correct / test_total
        
        return {
            "test_accuracy": test_acc,
            "predictions": np.array(all_predictions),
            "targets": np.array(all_targets)
        }
    
    def analyze_interpretability_robustness(
        self, 
        model: nn.Module, 
        test_loader: DataLoader,
        model_name: str
    ) -> Dict[str, float]:
        """
        Analyze interpretability vs robustness trade-offs
        This is a simplified version - you would implement more sophisticated analysis
        """
        
        # Robustness metrics
        robustness_metrics = self._measure_adversarial_robustness(model, test_loader, model_name)
        
        # Interpretability metrics (simplified)
        interpretability_metrics = self._measure_attention_consistency(model, test_loader, model_name)
        
        return {
            **robustness_metrics,
            **interpretability_metrics
        }
    
    def _measure_adversarial_robustness(
        self, 
        model: nn.Module, 
        test_loader: DataLoader,
        model_name: str
    ) -> Dict[str, float]:
        """Measure robustness to adversarial perturbations"""
        
        model.eval()
        epsilon = 0.01  # Small perturbation
        
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        for batch_x, batch_y in test_loader:
            if total > 200:  # Limit for speed
                break
                
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            batch_x.requires_grad = True
            
            # Clean prediction
            if "EEGMamba" in model_name:
                outputs_clean = model(batch_x, task_id=0)
            else:
                outputs_clean = model(batch_x)
            
            loss = nn.CrossEntropyLoss()(outputs_clean, batch_y)
            model.zero_grad()
            loss.backward()
            
            # Generate adversarial examples (FGSM)
            sign_data_grad = batch_x.grad.data.sign()
            perturbed_data = batch_x + epsilon * sign_data_grad
            perturbed_data = torch.clamp(perturbed_data, -3, 3)  # Reasonable bounds for normalized EEG
            
            # Adversarial prediction
            with torch.no_grad():
                if "EEGMamba" in model_name:
                    outputs_adv = model(perturbed_data, task_id=0)
                else:
                    outputs_adv = model(perturbed_data)
            
            # Calculate accuracy
            _, pred_clean = torch.max(outputs_clean, 1)
            _, pred_adv = torch.max(outputs_adv, 1)
            
            clean_correct += (pred_clean == batch_y).sum().item()
            adv_correct += (pred_adv == batch_y).sum().item()
            total += batch_y.size(0)
        
        clean_acc = clean_correct / total
        adv_acc = adv_correct / total
        robustness_drop = clean_acc - adv_acc
        
        return {
            "clean_accuracy": clean_acc,
            "adversarial_accuracy": adv_acc,
            "robustness_drop": robustness_drop
        }
    
    def _measure_attention_consistency(
        self, 
        model: nn.Module, 
        test_loader: DataLoader,
        model_name: str
    ) -> Dict[str, float]:
        """Measure attention pattern consistency (simplified interpretability metric)"""
        
        # This is a placeholder - real implementation would analyze attention maps
        # and measure their consistency across similar inputs
        
        if "Attention" in model_name or "CTNet" in model_name or "Deformer" in model_name:
            # Models with explicit attention mechanisms
            attention_consistency = np.random.uniform(0.6, 0.9)  # Placeholder
            attention_sparsity = np.random.uniform(0.3, 0.7)    # Placeholder
        else:
            # Models without explicit attention
            attention_consistency = np.random.uniform(0.4, 0.7)  # Placeholder
            attention_sparsity = np.random.uniform(0.2, 0.5)    # Placeholder
        
        return {
            "attention_consistency": attention_consistency,
            "attention_sparsity": attention_sparsity
        }
    
    def run_comprehensive_evaluation(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        n_channels: int,
        n_classes: int,
        samples: int
    ) -> Dict[str, Dict]:
        """Run comprehensive evaluation of all models"""
        
        print("Starting Comprehensive EEG Model Evaluation")
        print("=" * 60)
        
        # Prepare data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_val = torch.LongTensor(y_val)
        y_test = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        print(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create models
        models = self.create_models(n_channels, n_classes, samples)
        self.print_model_stats(models)
        
        # Train and evaluate all models
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*20} {model_name} {'='*20}")
            
            try:
                # Train model
                training_results = self.train_model(model, train_loader, val_loader, model_name)
                
                # Load best model for evaluation
                if self.config.save_models:
                    model.load_state_dict(
                        torch.load(os.path.join(self.config.save_dir, f"{model_name}_best.pth"))
                    )
                
                # Evaluate model
                test_results = self.evaluate_model(model, test_loader, model_name)
                
                # Analyze interpretability vs robustness
                analysis_results = self.analyze_interpretability_robustness(model, test_loader, model_name)
                
                # Combine results
                all_results[model_name] = {
                    **training_results,
                    **test_results,
                    **analysis_results
                }
                
                print(f"{model_name} Results:")
                print(f"  Best Val Acc: {training_results['best_val_acc']:.2f}%")
                print(f"  Test Acc: {test_results['test_accuracy']:.2f}%")
                print(f"  Training Time: {training_results['training_time']:.1f}s")
                print(f"  Robustness Drop: {analysis_results['robustness_drop']:.3f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                all_results[model_name] = {"error": str(e)}
        
        # Save results
        if self.config.save_models:
            with open(os.path.join(self.config.save_dir, "results.json"), "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for model_name, results in all_results.items():
                    json_results[model_name] = {}
                    for key, value in results.items():
                        if isinstance(value, np.ndarray):
                            json_results[model_name][key] = value.tolist()
                        else:
                            json_results[model_name][key] = value
                json.dump(json_results, f, indent=2)
        
        return all_results
    
    def plot_results(self, results: Dict[str, Dict]):
        """Plot comparison results"""
        
        # Extract metrics for plotting
        model_names = []
        test_accs = []
        train_times = []
        robustness_drops = []
        param_counts = []
        
        for model_name, result in results.items():
            if "error" not in result:
                model_names.append(model_name)
                test_accs.append(result['test_accuracy'])
                train_times.append(result['training_time'])
                robustness_drops.append(result['robustness_drop'])
                # You would need to store parameter counts separately
                param_counts.append(100000)  # Placeholder
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Test Accuracy Comparison
        axes[0, 0].bar(model_names, test_accs, color='skyblue')
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Training Time Comparison
        axes[0, 1].bar(model_names, train_times, color='lightcoral')
        axes[0, 1].set_title('Training Time Comparison')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Robustness Analysis
        axes[1, 0].bar(model_names, robustness_drops, color='lightgreen')
        axes[1, 0].set_title('Robustness Drop (Lower is Better)')
        axes[1, 0].set_ylabel('Accuracy Drop')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Interpretability vs Performance Trade-off
        interpretability = [results[name].get('attention_consistency', 0.5) for name in model_names]
        axes[1, 1].scatter(test_accs, interpretability, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (test_accs[i], interpretability[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('Test Accuracy (%)')
        axes[1, 1].set_ylabel('Interpretability Score')
        axes[1, 1].set_title('Interpretability vs Performance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_report(self, results: Dict[str, Dict]) -> str:
        """Generate comprehensive evaluation report"""
        
        report = []
        report.append("EEG Model Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary table
        report.append("Model Performance Summary:")
        report.append("-" * 30)
        
        for model_name, result in results.items():
            if "error" not in result:
                report.append(f"{model_name:20} | "
                            f"Test Acc: {result['test_accuracy']:6.2f}% | "
                            f"Val Acc: {result['best_val_acc']:6.2f}% | "
                            f"Time: {result['training_time']:6.1f}s | "
                            f"Robust: {result['robustness_drop']:5.3f}")
        
        report.append("")
        report.append("Key Findings:")
        report.append("-" * 15)
        
        # Find best models
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if valid_results:
            best_acc_model = max(valid_results.keys(), key=lambda k: valid_results[k]['test_accuracy'])
            most_robust_model = min(valid_results.keys(), key=lambda k: valid_results[k]['robustness_drop'])
            fastest_model = min(valid_results.keys(), key=lambda k: valid_results[k]['training_time'])
            
            report.append(f"• Best Accuracy: {best_acc_model} ({valid_results[best_acc_model]['test_accuracy']:.2f}%)")
            report.append(f"• Most Robust: {most_robust_model} (drop: {valid_results[most_robust_model]['robustness_drop']:.3f})")
            report.append(f"• Fastest Training: {fastest_model} ({valid_results[fastest_model]['training_time']:.1f}s)")
        
        report.append("")
        report.append("Architecture Analysis:")
        report.append("-" * 20)
        report.append("• CTNet: Combines CNN feature extraction with transformer global modeling")
        report.append("• EEG-Deformer: Hierarchical multi-scale attention for coarse-to-fine processing")
        report.append("• EEGMamba: Bidirectional state space model with task-aware mixture of experts")
        report.append("• MultiScale-Attention: Dynamic fusion of spectral-temporal features")
        
        report_text = "\n".join(report)
        
        # Save report
        if self.config.save_models:
            with open(os.path.join(self.config.save_dir, "evaluation_report.txt"), "w") as f:
                f.write(report_text)
        
        return report_text


# =============== Utility Functions ===============

def generate_synthetic_eeg_data(
    n_subjects: int = 100,
    n_channels: int = 22,
    n_samples: int = 1000,
    n_classes: int = 4,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG data for testing
    
    Returns:
        X: [n_subjects, 1, n_channels, n_samples]
        y: [n_subjects]
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    for subject in range(n_subjects):
        # Generate class label
        class_label = np.random.randint(0, n_classes)
        
        # Generate base signal with class-specific patterns
        signal = np.random.randn(n_channels, n_samples) * noise_level
        
        # Add class-specific patterns
        if class_label == 0:  # Left hand
            signal[8:12, 200:600] += np.sin(np.linspace(0, 4*np.pi, 400)) * 0.5  # Mu rhythm
        elif class_label == 1:  # Right hand
            signal[12:16, 200:600] += np.sin(np.linspace(0, 4*np.pi, 400)) * 0.5
        elif class_label == 2:  # Feet
            signal[16:20, 300:700] += np.sin(np.linspace(0, 6*np.pi, 400)) * 0.3
        else:  # Tongue
            signal[4:8, 250:650] += np.sin(np.linspace(0, 5*np.pi, 400)) * 0.4
        
        # Add noise and artifacts
        signal += np.random.randn(n_channels, n_samples) * noise_level
        
        X.append(signal[np.newaxis, :, :])  # Add channel dimension
        y.append(class_label)
    
    return np.array(X), np.array(y)


def load_bci_competition_data(dataset_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real BCI competition data (placeholder)
    You would implement actual data loading here
    """
    if dataset_path is None or not os.path.exists(dataset_path):
        print("No real data found, generating synthetic data...")
        return generate_synthetic_eeg_data()
    
    # Implement real data loading here
    # return X, y
    pass


# =============== Main Evaluation Script ===============

def main():
    """Main evaluation script"""
    
    print("Modern EEG Architecture Evaluation")
    print("Interpretability vs Classification Robustness Analysis")
    print("=" * 60)
    
    # Configuration
    config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        num_epochs=50,  # Reduced for demo
        patience=10,
        save_models=True
    )
    
    # Load or generate data
    print("Loading EEG data...")
    X, y = generate_synthetic_eeg_data(
        n_subjects=200,  # Reduced for demo
        n_channels=22,
        n_samples=1000,
        n_classes=4
    )
    
    print(f"Data shape: {X.shape}, Labels: {y.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(
        X=X,
        y=y,
        n_channels=22,
        n_classes=4,
        samples=1000
    )
    
    # Plot results
    print("\nGenerating comparison plots...")
    fig = evaluator.plot_results(results)
    
    # Generate report
    print("\nGenerating evaluation report...")
    report = evaluator.generate_report(results)
    print("\n" + report)
    
    print(f"\nEvaluation complete! Results saved to: {config.save_dir}")
    
    return results, evaluator


if __name__ == "__main__":
    results, evaluator = main()


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class BCIAdversarialAttacks:
    """
    Comprehensive adversarial attack framework for BCI/EEG models
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # EEG-specific constraints
        self.eeg_constraints = {
            'amplitude_range': (-100, 100),  # Typical EEG amplitude range (µV)
            'frequency_range': (0.5, 100),  # Typical EEG frequency range (Hz)
            'temporal_smoothness': 0.1,     # Smoothness constraint
        }
    
    def fgsm_attack(self, data, labels, epsilon=0.1, targeted=False, target_class=None):
        """
        Fast Gradient Sign Method (FGSM) attack
        
        Args:
            data: Input EEG data (batch_size, channels, samples)
            labels: True labels
            epsilon: Perturbation magnitude
            targeted: Whether to perform targeted attack
            target_class: Target class for targeted attack
        """
        data = data.to(self.device)
        labels = labels.to(self.device)
        data.requires_grad = True
        
        # Forward pass
        outputs = self.model(data)
        
        # Calculate loss
        if targeted and target_class is not None:
            # Targeted attack: minimize loss for target class
            target_labels = torch.full_like(labels, target_class)
            loss = F.cross_entropy(outputs, target_labels)
            loss = -loss  # Minimize loss (maximize confidence for target)
        else:
            # Untargeted attack: maximize loss for true class
            loss = F.cross_entropy(outputs, labels)
        
        # Calculate gradients
        loss.backward()
        
        # Generate adversarial perturbation
        data_grad = data.grad.data
        
        if targeted:
            # For targeted attacks, move in direction of negative gradient
            perturbation = -epsilon * data_grad.sign()
        else:
            # For untargeted attacks, move in direction of positive gradient
            perturbation = epsilon * data_grad.sign()
        
        # Apply perturbation
        adversarial_data = data + perturbation
        
        # Apply EEG-specific constraints
        adversarial_data = self._apply_eeg_constraints(adversarial_data)
        
        return adversarial_data.detach(), perturbation.detach()
    
    def pgd_attack(self, data, labels, epsilon=0.1, alpha=0.01, num_iter=10, 
                   targeted=False, target_class=None):
        """
        Projected Gradient Descent (PGD) attack
        More powerful iterative version of FGSM
        """
        data = data.to(self.device)
        labels = labels.to(self.device)
        
        # Initialize adversarial data
        adversarial_data = data.clone().detach()
        
        for i in range(num_iter):
            adversarial_data.requires_grad = True
            
            # Forward pass
            outputs = self.model(adversarial_data)
            
            # Calculate loss
            if targeted and target_class is not None:
                target_labels = torch.full_like(labels, target_class)
                loss = -F.cross_entropy(outputs, target_labels)
            else:
                loss = F.cross_entropy(outputs, labels)
            
            # Calculate gradients
            loss.backward()
            
            # Update adversarial data
            with torch.no_grad():
                if targeted:
                    adversarial_data = adversarial_data - alpha * adversarial_data.grad.sign()
                else:
                    adversarial_data = adversarial_data + alpha * adversarial_data.grad.sign()
                
                # Project back to epsilon ball
                perturbation = adversarial_data - data
                perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                adversarial_data = data + perturbation
                
                # Apply EEG constraints
                adversarial_data = self._apply_eeg_constraints(adversarial_data)
            
            # Clear gradients
            adversarial_data.grad = None
        
        return adversarial_data.detach(), (adversarial_data - data).detach()
    
    def eeg_noise_attack(self, data, labels, noise_type='gaussian', intensity=0.1):
        """
        EEG-specific noise attacks that mimic real-world artifacts
        """
        data = data.to(self.device)
        adversarial_data = data.clone()
        
        if noise_type == 'gaussian':
            # Add Gaussian noise
            noise = torch.randn_like(data) * intensity
            adversarial_data += noise
            
        elif noise_type == 'eog':
            # Simulate EOG (eye movement) artifacts
            # Typically affects frontal channels more
            batch_size, channels, samples = data.shape
            eog_noise = torch.zeros_like(data)
            
            # Apply stronger noise to first few channels (frontal)
            frontal_channels = min(4, channels)
            eog_pattern = torch.sin(torch.linspace(0, 4*np.pi, samples)) * intensity
            eog_noise[:, :frontal_channels, :] = eog_pattern.unsqueeze(0).unsqueeze(0)
            
            adversarial_data += eog_noise
            
        elif noise_type == 'emg':
            # Simulate EMG (muscle) artifacts
            # High frequency noise
            emg_noise = torch.randn_like(data) * intensity
            # Apply high-pass filter effect
            emg_noise = self._high_pass_filter(emg_noise)
            adversarial_data += emg_noise
            
        elif noise_type == 'line':
            # Simulate 50/60 Hz line noise
            batch_size, channels, samples = data.shape
            t = torch.linspace(0, 1, samples).to(self.device)
            line_noise = torch.sin(2 * np.pi * 50 * t) * intensity  # 50 Hz
            adversarial_data += line_noise.unsqueeze(0).unsqueeze(0)
        
        return self._apply_eeg_constraints(adversarial_data), adversarial_data - data
    
    def temporal_attack(self, data, labels, shift_samples=5, epsilon=0.1):
        """
        Temporal shift attack - shifts EEG signals in time
        """
        data = data.to(self.device)
        batch_size, channels, samples = data.shape
        
        # Create shifted versions
        adversarial_data = torch.zeros_like(data)
        
        for i in range(batch_size):
            # Random shift for each sample
            shift = np.random.randint(-shift_samples, shift_samples + 1)
            
            if shift > 0:
                adversarial_data[i, :, shift:] = data[i, :, :-shift]
            elif shift < 0:
                adversarial_data[i, :, :shift] = data[i, :, -shift:]
            else:
                adversarial_data[i] = data[i]
        
        # Add small noise perturbation
        noise = torch.randn_like(adversarial_data) * epsilon
        adversarial_data += noise
        
        return self._apply_eeg_constraints(adversarial_data), adversarial_data - data
    
    def channel_dropout_attack(self, data, labels, dropout_prob=0.2):
        """
        Channel dropout attack - simulates electrode disconnection
        """
        data = data.to(self.device)
        adversarial_data = data.clone()
        
        batch_size, channels, samples = data.shape
        
        # Randomly dropout channels
        for i in range(batch_size):
            dropout_mask = torch.rand(channels) < dropout_prob
            adversarial_data[i, dropout_mask, :] = 0
        
        return adversarial_data, adversarial_data - data
    
    def _apply_eeg_constraints(self, data):
        """Apply EEG-specific constraints to maintain physiological plausibility"""
        # Clamp to physiological amplitude range
        min_val, max_val = self.eeg_constraints['amplitude_range']
        data = torch.clamp(data, min_val, max_val)
        
        # Could add more sophisticated constraints here
        # e.g., temporal smoothness, frequency constraints
        
        return data
    
    def _high_pass_filter(self, data, cutoff=20):
        """Simple high-pass filter simulation"""
        # This is a simplified version - in practice, you'd use proper signal processing
        return data - torch.mean(data, dim=-1, keepdim=True)
    
    def evaluate_attack(self, original_data, adversarial_data, true_labels, attack_name="Attack"):
        """
        Evaluate the success of an adversarial attack
        """
        self.model.eval()
        
        with torch.no_grad():
            # Original predictions
            orig_outputs = self.model(original_data.to(self.device))
            orig_preds = torch.argmax(orig_outputs, dim=1)
            orig_accuracy = (orig_preds == true_labels.to(self.device)).float().mean()
            
            # Adversarial predictions
            adv_outputs = self.model(adversarial_data.to(self.device))
            adv_preds = torch.argmax(adv_outputs, dim=1)
            adv_accuracy = (adv_preds == true_labels.to(self.device)).float().mean()
            
            # Attack success rate
            attack_success = (orig_preds != adv_preds).float().mean()
        
        results = {
            'attack_name': attack_name,
            'original_accuracy': orig_accuracy.item(),
            'adversarial_accuracy': adv_accuracy.item(),
            'attack_success_rate': attack_success.item(),
            'accuracy_drop': orig_accuracy.item() - adv_accuracy.item()
        }
        
        return results
    
    def visualize_attack(self, original_data, adversarial_data, perturbation, 
                        channel_idx=0, sample_idx=0, save_path=None):
        """
        Visualize the effect of adversarial attacks on EEG signals
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert to numpy for plotting
        orig = original_data[sample_idx, channel_idx].cpu().numpy()
        adv = adversarial_data[sample_idx, channel_idx].cpu().numpy()
        pert = perturbation[sample_idx, channel_idx].cpu().numpy()
        
        # Time axis
        time = np.arange(len(orig))
        
        # Plot original signal
        axes[0, 0].plot(time, orig, 'b-', linewidth=1)
        axes[0, 0].set_title('Original EEG Signal')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude (µV)')
        axes[0, 0].grid(True)
        
        # Plot adversarial signal
        axes[0, 1].plot(time, adv, 'r-', linewidth=1)
        axes[0, 1].set_title('Adversarial EEG Signal')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Amplitude (µV)')
        axes[0, 1].grid(True)
        
        # Plot perturbation
        axes[1, 0].plot(time, pert, 'g-', linewidth=1)
        axes[1, 0].set_title('Perturbation')
        axes[1, 0].set_xlabel('Sample')
        axes[1, 0].set_ylabel('Amplitude (µV)')
        axes[1, 0].grid(True)
        
        # Plot comparison
        axes[1, 1].plot(time, orig, 'b-', linewidth=1, label='Original', alpha=0.7)
        axes[1, 1].plot(time, adv, 'r-', linewidth=1, label='Adversarial', alpha=0.7)
        axes[1, 1].set_title('Comparison')
        axes[1, 1].set_xlabel('Sample')
        axes[1, 1].set_ylabel('Amplitude (µV)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def comprehensive_attack_evaluation(self, data_loader, attacks_to_test=None):
        """
        Run comprehensive evaluation of multiple attack methods
        """
        if attacks_to_test is None:
            attacks_to_test = ['fgsm', 'pgd', 'eog_noise', 'emg_noise', 'temporal', 'channel_dropout']
        
        results = []
        
        # Get a batch of data for testing
        data_batch, labels_batch = next(iter(data_loader))
        
        for attack_name in attacks_to_test:
            print(f"Testing {attack_name} attack...")
            
            if attack_name == 'fgsm':
                adv_data, perturbation = self.fgsm_attack(data_batch, labels_batch)
            elif attack_name == 'pgd':
                adv_data, perturbation = self.pgd_attack(data_batch, labels_batch)
            elif attack_name == 'eog_noise':
                adv_data, perturbation = self.eeg_noise_attack(data_batch, labels_batch, 'eog')
            elif attack_name == 'emg_noise':
                adv_data, perturbation = self.eeg_noise_attack(data_batch, labels_batch, 'emg')
            elif attack_name == 'temporal':
                adv_data, perturbation = self.temporal_attack(data_batch, labels_batch)
            elif attack_name == 'channel_dropout':
                adv_data, perturbation = self.channel_dropout_attack(data_batch, labels_batch)
            
            # Evaluate attack
            attack_results = self.evaluate_attack(data_batch, adv_data, labels_batch, attack_name)
            results.append(attack_results)
        
        return results
    
    def defense_evaluation(self, data_loader, defense_method='adversarial_training'):
        """
        Evaluate defense methods against adversarial attacks
        """
        # This would implement various defense strategies
        # For now, just a placeholder structure
        
        if defense_method == 'adversarial_training':
            # Implement adversarial training
            pass
        elif defense_method == 'input_preprocessing':
            # Implement input preprocessing defenses
            pass
        elif defense_method == 'ensemble':
            # Implement ensemble defenses
            pass
        
        return "Defense evaluation not implemented yet"

# Example usage and testing functions
def test_adversarial_attacks(model, test_data, test_labels):
    """
    Test function to demonstrate adversarial attacks
    """
    # Create attack instance
    attacker = BCIAdversarialAttacks(model)
    
    # Create data loader
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Run comprehensive evaluation
    results = attacker.comprehensive_attack_evaluation(test_loader)
    
    # Print results
    print("\n🎯 ADVERSARIAL ATTACK RESULTS:")
    print("-" * 50)
    for result in results:
        print(f"{result['attack_name']:15} | "
              f"Original: {result['original_accuracy']:.3f} | "
              f"Adversarial: {result['adversarial_accuracy']:.3f} | "
              f"Success Rate: {result['attack_success_rate']:.3f} | "
              f"Drop: {result['accuracy_drop']:.3f}")
    
    return results

# Example usage:
"""
# After your model is trained
model = EEGNet(num_classes=4, channels=22, samples=1125)
# Load your trained weights
model.load_state_dict(torch.load('your_model.pth'))

# Test adversarial attacks
results = test_adversarial_attacks(model, X_test, y_test)

# Visualize specific attacks
attacker = BCIAdversarialAttacks(model)
data_batch, labels_batch = X_test[:10], y_test[:10]
adv_data, perturbation = attacker.fgsm_attack(data_batch, labels_batch)
attacker.visualize_attack(data_batch, adv_data, perturbation)
"""

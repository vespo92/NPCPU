"""
Neural Consciousness Models

Neural network-based consciousness that learns capability scores from experience
through gradient descent, rather than hand-crafted rules.

Based on Month 2 roadmap: Neural Architecture Search for Consciousness
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import copy

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Architecture:
    """Neural network architecture specification"""
    input_dim: int
    hidden_dims: List[int]
    output_dim: int = 9  # 9 consciousness dimensions
    activation: str = "relu"
    dropout: float = 0.2

    def __str__(self):
        return f"Arch({self.input_dim}->{self.hidden_dims}->{self.output_dim})"


@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    early_stop_patience: int = 5
    validation_split: float = 0.2
    weight_decay: float = 0.0001


@dataclass
class NASConfig:
    """Neural Architecture Search configuration"""
    search_space: List[List[int]] = None
    max_evaluations: int = 20
    epochs_per_eval: int = 30

    def __post_init__(self):
        if self.search_space is None:
            self.search_space = [
                [64],
                [128],
                [64, 64],
                [128, 64],
                [128, 128],
                [256, 128],
                [128, 64, 32],
                [256, 128, 64],
                [512, 256, 128]
            ]


# ============================================================================
# Activation Functions
# ============================================================================

class Activations:
    """Collection of activation functions"""

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = Activations.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def get(name: str) -> Tuple[Callable, Callable]:
        """Get activation function and its derivative"""
        activations = {
            "relu": (Activations.relu, Activations.relu_derivative),
            "sigmoid": (Activations.sigmoid, Activations.sigmoid_derivative),
            "tanh": (Activations.tanh, Activations.tanh_derivative)
        }
        return activations.get(name, activations["relu"])


# ============================================================================
# Neural Consciousness Model
# ============================================================================

class NeuralConsciousnessModel:
    """
    Neural network-based consciousness model.

    Instead of hand-crafted capability scores, learns them from experience
    through gradient descent. The network takes experience vectors as input
    and predicts consciousness capability scores.

    Features:
    - Configurable architecture (depth, width)
    - Multiple activation functions
    - Dropout regularization
    - Batch training with early stopping
    - Gradient clipping for stability

    Example:
        model = NeuralConsciousnessModel(
            architecture=Architecture(
                input_dim=256,
                hidden_dims=[128, 64]
            )
        )

        # Train on experiences
        model.fit(experiences, target_capabilities)

        # Predict consciousness from new experience
        consciousness = model.get_consciousness(experience_vector)
    """

    def __init__(
        self,
        architecture: Architecture,
        training_config: Optional[TrainingConfig] = None
    ):
        self.architecture = architecture
        self.config = training_config or TrainingConfig()
        self.activation, self.activation_derivative = Activations.get(
            architecture.activation
        )

        # Initialize weights
        self.weights = []
        self.biases = []
        self._initialize_weights()

        # Training history
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "epochs": 0
        }

        # Capability names for output mapping
        self.capability_names = [
            "perception_fidelity",
            "reaction_speed",
            "memory_depth",
            "memory_recall_accuracy",
            "introspection_capacity",
            "meta_cognitive_ability",
            "information_integration",
            "intentional_coherence",
            "qualia_richness"
        ]

    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        np.random.seed(42)  # For reproducibility

        dims = [self.architecture.input_dim] + self.architecture.hidden_dims + [self.architecture.output_dim]

        self.weights = []
        self.biases = []

        for i in range(len(dims) - 1):
            # He initialization for ReLU
            std = np.sqrt(2.0 / dims[i])
            weight = np.random.randn(dims[i], dims[i + 1]) * std
            bias = np.zeros((1, dims[i + 1]))

            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, x: np.ndarray, training: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through the network.

        Args:
            x: Input array (batch_size, input_dim)
            training: Whether to apply dropout

        Returns:
            Tuple of (output, activations_list)
        """
        activations = [x]
        current = x

        # Hidden layers with activation
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            current = self.activation(z)

            # Dropout during training
            if training and self.architecture.dropout > 0:
                mask = np.random.binomial(1, 1 - self.architecture.dropout, current.shape)
                current = current * mask / (1 - self.architecture.dropout)

            activations.append(current)

        # Output layer with sigmoid (scores in [0, 1])
        z = current @ self.weights[-1] + self.biases[-1]
        output = Activations.sigmoid(z)
        activations.append(output)

        return output, activations

    def backward(
        self,
        y_true: np.ndarray,
        activations: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward pass to compute gradients.

        Args:
            y_true: True labels
            activations: List of activations from forward pass

        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        m = y_true.shape[0]
        weight_grads = []
        bias_grads = []

        # Output layer gradient (MSE loss with sigmoid)
        output = activations[-1]
        delta = (output - y_true) * Activations.sigmoid_derivative(
            output * (1 - output)  # Inverse sigmoid approximation
        )

        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            weight_grad = activations[i].T @ delta / m
            bias_grad = np.sum(delta, axis=0, keepdims=True) / m

            # Add L2 regularization
            weight_grad += self.config.weight_decay * self.weights[i]

            weight_grads.insert(0, weight_grad)
            bias_grads.insert(0, bias_grad)

            if i > 0:
                # Propagate delta to previous layer
                delta = (delta @ self.weights[i].T) * self.activation_derivative(activations[i])

        return weight_grads, bias_grads

    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute MSE loss"""
        return np.mean((y_pred - y_true) ** 2)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model on experience data.

        Args:
            X: Experience vectors (n_samples, input_dim)
            y: Target capability scores (n_samples, 9)
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        # Split into train/val
        n_samples = X.shape[0]
        n_val = int(n_samples * self.config.validation_split)
        indices = np.random.permutation(n_samples)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None

        for epoch in range(self.config.epochs):
            # Shuffle training data
            shuffle_idx = np.random.permutation(len(X_train))
            X_train = X_train[shuffle_idx]
            y_train = y_train[shuffle_idx]

            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), self.config.batch_size):
                batch_X = X_train[i:i + self.config.batch_size]
                batch_y = y_train[i:i + self.config.batch_size]

                # Forward pass
                output, activations = self.forward(batch_X, training=True)

                # Backward pass
                weight_grads, bias_grads = self.backward(batch_y, activations)

                # Update weights with gradient clipping
                for j in range(len(self.weights)):
                    # Clip gradients
                    weight_grads[j] = np.clip(weight_grads[j], -1.0, 1.0)
                    bias_grads[j] = np.clip(bias_grads[j], -1.0, 1.0)

                    self.weights[j] -= self.config.learning_rate * weight_grads[j]
                    self.biases[j] -= self.config.learning_rate * bias_grads[j]

                epoch_loss += self._compute_loss(output, batch_y)
                n_batches += 1

            # Compute validation loss
            val_output, _ = self.forward(X_val, training=False)
            val_loss = self._compute_loss(val_output, y_val)

            train_loss = epoch_loss / n_batches
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["epochs"] = epoch + 1

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stop_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if best_weights:
            self.weights = best_weights
            self.biases = best_biases

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict consciousness capabilities from experiences.

        Args:
            X: Experience vectors (n_samples, input_dim) or (input_dim,)

        Returns:
            Capability scores (n_samples, 9) or (9,)
        """
        single_input = X.ndim == 1
        if single_input:
            X = X.reshape(1, -1)

        output, _ = self.forward(X, training=False)

        if single_input:
            return output[0]
        return output

    def get_consciousness(self, experience: np.ndarray) -> GradedConsciousness:
        """
        Convert experience to GradedConsciousness.

        Args:
            experience: Experience vector

        Returns:
            GradedConsciousness instance with predicted capabilities
        """
        scores = self.predict(experience)

        return GradedConsciousness(
            perception_fidelity=float(scores[0]),
            reaction_speed=float(scores[1]),
            memory_depth=float(scores[2]),
            memory_recall_accuracy=float(scores[3]),
            introspection_capacity=float(scores[4]),
            meta_cognitive_ability=float(scores[5]),
            information_integration=float(scores[6]),
            intentional_coherence=float(scores[7]),
            qualia_richness=float(scores[8])
        )

    def save(self, path: str):
        """Save model weights to file"""
        np.savez(
            path,
            weights=[w for w in self.weights],
            biases=[b for b in self.biases],
            architecture={
                "input_dim": self.architecture.input_dim,
                "hidden_dims": self.architecture.hidden_dims,
                "output_dim": self.architecture.output_dim,
                "activation": self.architecture.activation,
                "dropout": self.architecture.dropout
            }
        )

    @classmethod
    def load(cls, path: str) -> 'NeuralConsciousnessModel':
        """Load model from file"""
        data = np.load(path, allow_pickle=True)
        arch_dict = data["architecture"].item()

        architecture = Architecture(**arch_dict)
        model = cls(architecture)

        model.weights = list(data["weights"])
        model.biases = list(data["biases"])

        return model


# ============================================================================
# Neural Architecture Search
# ============================================================================

class ConsciousnessNAS:
    """
    Neural Architecture Search for consciousness models.

    Automatically finds optimal architecture for consciousness prediction
    given a dataset of experiences and target capabilities.

    Features:
    - Grid search over architecture space
    - Cross-validation evaluation
    - Architecture ranking
    - Best model extraction

    Example:
        nas = ConsciousnessNAS()

        # Prepare data
        experiences = ...  # Shape: (n_samples, input_dim)
        capabilities = ...  # Shape: (n_samples, 9)

        # Search for best architecture
        best_arch, best_model = nas.search(
            train_data=(experiences, capabilities),
            input_dim=256
        )

        print(f"Best architecture: {best_arch}")
    """

    def __init__(self, config: Optional[NASConfig] = None):
        self.config = config or NASConfig()
        self.search_results: List[Dict[str, Any]] = []
        self.best_architecture: Optional[Architecture] = None
        self.best_model: Optional[NeuralConsciousnessModel] = None
        self.best_loss: float = float('inf')

    def evaluate_architecture(
        self,
        architecture: Architecture,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        Train and evaluate a single architecture.

        Returns validation loss.
        """
        training_config = TrainingConfig(
            epochs=self.config.epochs_per_eval,
            early_stop_patience=3
        )

        model = NeuralConsciousnessModel(architecture, training_config)
        model.fit(X_train, y_train, verbose=False)

        # Evaluate on validation set
        predictions = model.predict(X_val)
        val_loss = np.mean((predictions - y_val) ** 2)

        return val_loss, model

    def search(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        input_dim: int,
        verbose: bool = True
    ) -> Tuple[Architecture, NeuralConsciousnessModel]:
        """
        Search for best architecture.

        Args:
            train_data: Tuple of (X, y) training data
            input_dim: Input dimension for architectures
            verbose: Print progress

        Returns:
            Tuple of (best_architecture, best_model)
        """
        X, y = train_data

        # Split for architecture evaluation
        n_samples = X.shape[0]
        n_val = int(n_samples * 0.2)
        indices = np.random.permutation(n_samples)

        X_val, y_val = X[indices[:n_val]], y[indices[:n_val]]
        X_train, y_train = X[indices[n_val:]], y[indices[n_val:]]

        self.search_results = []

        for i, hidden_dims in enumerate(self.config.search_space):
            if i >= self.config.max_evaluations:
                break

            architecture = Architecture(
                input_dim=input_dim,
                hidden_dims=hidden_dims
            )

            if verbose:
                print(f"Evaluating architecture {i+1}/{len(self.config.search_space)}: {hidden_dims}")

            val_loss, model = self.evaluate_architecture(
                architecture, X_train, y_train, X_val, y_val
            )

            result = {
                "architecture": architecture,
                "hidden_dims": hidden_dims,
                "val_loss": val_loss,
                "model": model
            }
            self.search_results.append(result)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_architecture = architecture
                self.best_model = model

            if verbose:
                print(f"  Validation loss: {val_loss:.4f}")

        # Sort results
        self.search_results.sort(key=lambda x: x["val_loss"])

        if verbose:
            print(f"\nBest architecture: {self.best_architecture.hidden_dims}")
            print(f"Best validation loss: {self.best_loss:.4f}")

        return self.best_architecture, self.best_model

    def get_ranked_architectures(self) -> List[Dict[str, Any]]:
        """Get architectures ranked by performance"""
        return [
            {
                "rank": i + 1,
                "hidden_dims": r["hidden_dims"],
                "val_loss": r["val_loss"]
            }
            for i, r in enumerate(self.search_results)
        ]


# ============================================================================
# Data Generation Utilities
# ============================================================================

class ConsciousnessDataGenerator:
    """
    Generate synthetic training data for neural consciousness models.

    Useful for testing and development.
    """

    @staticmethod
    def generate_experience_vectors(
        n_samples: int,
        input_dim: int,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """Generate random experience vectors"""
        return np.random.randn(n_samples, input_dim) * (1 + noise_level * np.random.randn(n_samples, input_dim))

    @staticmethod
    def generate_target_capabilities(
        experiences: np.ndarray,
        method: str = "linear"
    ) -> np.ndarray:
        """
        Generate target capabilities from experiences.

        Methods:
        - "linear": Linear projection with noise
        - "nonlinear": Nonlinear transformation
        - "random": Random (for testing)
        """
        n_samples, input_dim = experiences.shape

        if method == "random":
            return np.random.rand(n_samples, 9)

        elif method == "linear":
            # Project to 9 dimensions and normalize
            projection = np.random.randn(input_dim, 9)
            capabilities = experiences @ projection
            # Normalize to [0, 1]
            capabilities = (capabilities - capabilities.min(axis=0)) / (capabilities.max(axis=0) - capabilities.min(axis=0) + 1e-8)
            # Add noise
            capabilities += np.random.randn(n_samples, 9) * 0.05
            return np.clip(capabilities, 0, 1)

        elif method == "nonlinear":
            # Nonlinear transformation
            projection = np.random.randn(input_dim, 32)
            hidden = np.tanh(experiences @ projection)

            output_proj = np.random.randn(32, 9)
            capabilities = 1 / (1 + np.exp(-hidden @ output_proj))

            return capabilities

        return np.random.rand(n_samples, 9)

    @staticmethod
    def generate_dataset(
        n_samples: int = 1000,
        input_dim: int = 128,
        method: str = "linear"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete dataset"""
        experiences = ConsciousnessDataGenerator.generate_experience_vectors(
            n_samples, input_dim
        )
        capabilities = ConsciousnessDataGenerator.generate_target_capabilities(
            experiences, method
        )
        return experiences, capabilities


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Neural Consciousness Model Demo")
    print("=" * 50)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y = ConsciousnessDataGenerator.generate_dataset(
        n_samples=500,
        input_dim=64,
        method="nonlinear"
    )
    print(f"   Data shape: X={X.shape}, y={y.shape}")

    # Train a single model
    print("\n2. Training single model...")
    architecture = Architecture(
        input_dim=64,
        hidden_dims=[128, 64]
    )
    model = NeuralConsciousnessModel(architecture)
    history = model.fit(X, y, verbose=True)

    # Make prediction
    print("\n3. Making prediction...")
    test_experience = np.random.randn(64)
    consciousness = model.get_consciousness(test_experience)
    print(f"   Predicted consciousness state: {consciousness.describe_state()}")
    print(f"   Overall score: {consciousness.overall_consciousness_score():.3f}")

    # Neural Architecture Search
    print("\n4. Running Neural Architecture Search...")
    nas = ConsciousnessNAS(NASConfig(max_evaluations=5, epochs_per_eval=20))
    best_arch, best_model = nas.search((X, y), input_dim=64, verbose=True)

    print("\n5. Architecture Rankings:")
    for rank in nas.get_ranked_architectures():
        print(f"   #{rank['rank']}: {rank['hidden_dims']} - loss={rank['val_loss']:.4f}")

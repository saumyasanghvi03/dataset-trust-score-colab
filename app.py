#!/usr/bin/env python3
"""
Dataset Trust Score Application for CIFAR-10

This script loads the CIFAR-10 dataset, allows selection of clean or poisoned subsets,
and runs the Dataset Trust Score logic in standalone mode, outputting verdict and
confidence to the console.
"""

import os
import sys
import numpy as np
import argparse
from typing import Tuple, Optional

# Import required libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as e:
    print(f"Error: Required libraries not found. Please install tensorflow: {e}")
    sys.exit(1)

class DatasetTrustScorer:
    """
    Dataset Trust Score implementation for evaluating dataset trustworthiness.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or 'cifar10_model.h5'
        self.model = None
        self.num_classes = 10
        self.img_shape = (32, 32, 3)
        
    def load_cifar10_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess CIFAR-10 dataset.
        
        Returns:
            Tuple of (x_train, y_train, x_test, y_test)
        """
        print("Loading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        return x_train, y_train, x_test, y_test
    
    def create_poisoned_subset(self, x_data: np.ndarray, y_data: np.ndarray, 
                             poison_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a poisoned subset by adding trigger patterns and changing labels.
        
        Args:
            x_data: Input images
            y_data: True labels
            poison_rate: Fraction of data to poison
            
        Returns:
            Tuple of (poisoned_x, poisoned_y)
        """
        print(f"Creating poisoned subset with {poison_rate*100}% poison rate...")
        
        x_poisoned = x_data.copy()
        y_poisoned = y_data.copy()
        
        num_samples = len(x_data)
        num_poison = int(num_samples * poison_rate)
        poison_indices = np.random.choice(num_samples, num_poison, replace=False)
        
        # Add simple trigger pattern (white square in bottom right corner)
        trigger_pattern = np.ones((4, 4, 3))
        
        for idx in poison_indices:
            # Add trigger pattern
            x_poisoned[idx, -4:, -4:, :] = trigger_pattern
            # Change label to target class (class 0)
            y_poisoned[idx] = to_categorical([0], self.num_classes)[0]
        
        print(f"Poisoned {num_poison} samples out of {num_samples}")
        return x_poisoned, y_poisoned
    
    def build_model(self) -> models.Model:
        """
        Build a CNN model for CIFAR-10 classification.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.img_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_or_load_model(self, x_train: np.ndarray, y_train: np.ndarray, 
                           x_test: np.ndarray, y_test: np.ndarray, 
                           force_retrain: bool = False) -> models.Model:
        """
        Load existing model or train a new one if necessary.
        
        Args:
            x_train, y_train: Training data
            x_test, y_test: Test data
            force_retrain: Force retraining even if model exists
            
        Returns:
            Trained Keras model
        """
        if os.path.exists(self.model_path) and not force_retrain:
            print(f"Loading existing model from {self.model_path}...")
            try:
                self.model = keras.models.load_model(self.model_path)
                return self.model
            except Exception as e:
                print(f"Failed to load model: {e}. Training new model...")
        
        print("Training new model...")
        self.model = self.build_model()
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            x_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save the model
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return self.model
    
    def calculate_trust_score(self, x_clean: np.ndarray, y_clean: np.ndarray,
                            x_suspicious: np.ndarray, y_suspicious: np.ndarray) -> Tuple[float, str]:
        """
        Calculate the Dataset Trust Score by comparing model performance
        on clean vs suspicious data.
        
        Args:
            x_clean, y_clean: Clean reference dataset
            x_suspicious, y_suspicious: Suspicious dataset to evaluate
            
        Returns:
            Tuple of (confidence_score, verdict)
        """
        print("\nCalculating Dataset Trust Score...")
        
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        # Evaluate on clean data
        clean_loss, clean_accuracy = self.model.evaluate(x_clean, y_clean, verbose=0)
        print(f"Clean data - Loss: {clean_loss:.4f}, Accuracy: {clean_accuracy:.4f}")
        
        # Evaluate on suspicious data
        suspicious_loss, suspicious_accuracy = self.model.evaluate(x_suspicious, y_suspicious, verbose=0)
        print(f"Suspicious data - Loss: {suspicious_loss:.4f}, Accuracy: {suspicious_accuracy:.4f}")
        
        # Calculate trust metrics
        accuracy_diff = abs(clean_accuracy - suspicious_accuracy)
        loss_diff = abs(clean_loss - suspicious_loss)
        
        # Simple trust score calculation
        # Higher differences suggest more suspicious data
        trust_score = 1.0 - min(1.0, (accuracy_diff * 2 + loss_diff * 0.5))
        
        # Determine verdict based on thresholds
        if trust_score > 0.8:
            verdict = "TRUSTED"
        elif trust_score > 0.6:
            verdict = "SUSPICIOUS"
        else:
            verdict = "UNTRUSTED"
        
        return trust_score, verdict
    
    def run_analysis(self, dataset_type: str = "clean", poison_rate: float = 0.1, 
                    force_retrain: bool = False) -> None:
        """
        Run the complete dataset trust analysis.
        
        Args:
            dataset_type: "clean" or "poisoned"
            poison_rate: Rate of poisoning if using poisoned dataset
            force_retrain: Force model retraining
        """
        print("=" * 60)
        print("Dataset Trust Score Analysis")
        print("=" * 60)
        
        # Load CIFAR-10 data
        x_train, y_train, x_test, y_test = self.load_cifar10_data()
        
        # Create clean reference (subset of original test data)
        clean_size = min(1000, len(x_test) // 2)
        x_clean_ref = x_test[:clean_size]
        y_clean_ref = y_test[:clean_size]
        
        # Prepare analysis dataset based on type
        if dataset_type.lower() == "poisoned":
            print(f"\nAnalyzing POISONED dataset (poison rate: {poison_rate})")
            x_analysis = x_test[clean_size:clean_size*2]
            y_analysis = y_test[clean_size:clean_size*2]
            x_analysis, y_analysis = self.create_poisoned_subset(x_analysis, y_analysis, poison_rate)
        else:
            print("\nAnalyzing CLEAN dataset")
            x_analysis = x_test[clean_size:clean_size*2]
            y_analysis = y_test[clean_size:clean_size*2]
        
        # Train or load model
        self.train_or_load_model(x_train, y_train, x_test, y_test, force_retrain)
        
        # Calculate trust score
        confidence, verdict = self.calculate_trust_score(x_clean_ref, y_clean_ref, x_analysis, y_analysis)
        
        # Output results
        print("\n" + "=" * 60)
        print("DATASET TRUST ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Dataset Type Analyzed: {dataset_type.upper()}")
        if dataset_type.lower() == "poisoned":
            print(f"Poison Rate: {poison_rate*100:.1f}%")
        print(f"Trust Score: {confidence:.4f}")
        print(f"Verdict: {verdict}")
        print("\nInterpretation:")
        if verdict == "TRUSTED":
            print("- The dataset appears to be clean and trustworthy")
        elif verdict == "SUSPICIOUS":
            print("- The dataset shows some anomalies and should be reviewed")
        else:
            print("- The dataset appears to be compromised or untrustworthy")
        print("=" * 60)

def main():
    """
    Main function to run the Dataset Trust Score application.
    """
    parser = argparse.ArgumentParser(description='Dataset Trust Score for CIFAR-10')
    parser.add_argument('--dataset-type', choices=['clean', 'poisoned'], default='clean',
                       help='Type of dataset to analyze (default: clean)')
    parser.add_argument('--poison-rate', type=float, default=0.1,
                       help='Poison rate for poisoned dataset (default: 0.1)')
    parser.add_argument('--model-path', type=str, default='cifar10_model.h5',
                       help='Path to save/load the model (default: cifar10_model.h5)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining of the model')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        # Initialize the trust scorer
        scorer = DatasetTrustScorer(model_path=args.model_path)
        
        # Run the analysis
        scorer.run_analysis(
            dataset_type=args.dataset_type,
            poison_rate=args.poison_rate,
            force_retrain=args.force_retrain
        )
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

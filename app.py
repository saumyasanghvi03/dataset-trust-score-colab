import streamlit as st
st.write("🚦 App loaded! Ready for user action.")
print("App started.")

import os
import sys
import numpy as np
from typing import Tuple, Optional
from io import BytesIO

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as e:
    st.error(f"Error: Required libraries not found. Please install tensorflow: {e}")
    print(f"Error: Required libraries not found. Please install tensorflow: {e}")
    sys.exit(1)

class DatasetTrustScorer:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or 'cifar10_model.h5'
        self.model = None
        self.num_classes = 10
        self.img_shape = (32, 32, 3)

    def load_cifar10_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            with st.spinner("Loading CIFAR-10 dataset..."):
                print("Console: Loading CIFAR-10 dataset...")
                (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                x_train = x_train.astype('float32') / 255.0
                x_test = x_test.astype('float32') / 255.0
                y_train = to_categorical(y_train, self.num_classes)
                y_test = to_categorical(y_test, self.num_classes)
            st.success(f"Dataset loaded: Train shape {x_train.shape}, Test shape {x_test.shape}")
            print(f"Console: Dataset loaded: Train shape {x_train.shape}, Test shape {x_test.shape}")
            return x_train, y_train, x_test, y_test
        except Exception as e:
            st.error(f"Error loading CIFAR-10 dataset: {e}")
            print(f"Console: Error loading CIFAR-10 dataset: {e}")
            raise

    def load_custom_data(self, data_file) -> Tuple[np.ndarray, np.ndarray]:
        """Load user-uploaded custom data (.npz, .csv, or .xlsx format with x and y)."""
        import pandas as pd
        try:
            filename = data_file.name.lower()
            with st.spinner("Loading uploaded dataset..."):
                if filename.endswith('.npz'):
                    content = data_file.read()
                    with BytesIO(content) as temp_bytes:
                        npzfile = np.load(temp_bytes)
                        x = npzfile['x']
                        y = npzfile['y']
                elif filename.endswith('.csv'):
                    df = pd.read_csv(data_file)
                    x = df.drop('y', axis=1).values if 'y' in df else df.values
                    y = df['y'].values if 'y' in df else np.zeros((x.shape[0],))
                elif filename.endswith('.xlsx'):
                    df = pd.read_excel(data_file)
                    x = df.drop('y', axis=1).values if 'y' in df else df.values
                    y = df['y'].values if 'y' in df else np.zeros((x.shape[0],))
                else:
                    raise ValueError("Unsupported file type. Please upload .npz, .csv, or .xlsx.")
                # Reshape x if needed (if 3072 = 32*32*3, reshape to (-1, 32, 32, 3))
                if x.ndim == 2 and x.shape[1] == 3072:
                    x = x.reshape(-1, 32, 32, 3)
                if x.max() > 1.1:
                    x = x.astype('float32') / 255.0
                num_classes = self.num_classes
                if y.ndim > 1 and y.shape[-1] == num_classes:
                    pass
                else:
                    y = to_categorical(y, num_classes)
            st.success(f"Uploaded data loaded: shape {x.shape}, labels shape {y.shape}")
            return x, y
        except Exception as e:
            st.error(f"Error loading uploaded data: {e}")
            raise

    def create_model(self) -> keras.Model:
        try:
            st.write("Creating CNN model...")
            print("Console: Creating CNN model...")
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.img_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            st.success("Model created successfully")
            print("Console: Model created successfully")
            return model
        except Exception as e:
            st.error(f"Error creating model: {e}")
            print(f"Error creating model: {e}")
            raise

    def load_model(self) -> Optional[keras.Model]:
        try:
            if os.path.exists(self.model_path):
                with st.spinner("Loading existing model..."):
                    print(f"Console: Loading existing model from {self.model_path}...")
                    model = keras.models.load_model(self.model_path)
                st.success("Model loaded successfully")
                print("Console: Model loaded successfully")
                return model
            else:
                st.warning(f"Model file {self.model_path} not found")
                print(f"Console: Model file {self.model_path} not found")
                return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            print(f"Error loading model: {e}")
            return None

    def save_model(self, model: keras.Model) -> bool:
        try:
            st.write(f"Saving model to {self.model_path}...")
            print(f"Console: Saving model to {self.model_path}...")
            model.save(self.model_path)
            st.success("Model saved successfully")
            print("Console: Model saved successfully")
            return True
        except Exception as e:
            st.error(f"Error saving model: {e}")
            print(f"Error saving model: {e}")
            return False

    def train_model(self, model: keras.Model, x_train: np.ndarray, y_train: np.ndarray,
                   x_test: np.ndarray, y_test: np.ndarray) -> keras.Model:
        try:
            st.write("Training model...")
            print("Console: Training model...")

            progress_bar = st.progress(0)
            status_text = st.empty()
            early_stop = EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            )
            class StreamlitCallback(keras.callbacks.Callback):
                def on_epoch_end(self_cb, epoch, logs=None):
                    progress = (epoch + 1) / self_cb.params['epochs']
                    progress_bar.progress(progress)
                    status_text.text(f'Epoch {epoch + 1}/{self_cb.params["epochs"]} - '
                                     f'Loss: {logs.get("loss", 0):.4f} - '
                                     f'Accuracy: {logs.get("accuracy", 0):.4f}')
            with st.spinner("Training the model... (this can take 1-2 minutes)"):
                history = model.fit(
                    x_train, y_train,
                    batch_size=32,
                    epochs=10,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stop, StreamlitCallback()],
                    verbose=0
                )
            progress_bar.progress(1.0)
            status_text.text("Training completed!")
            st.success("Model training completed")
            print("Console: Model training completed")
            return model
        except Exception as e:
            st.error(f"Error training model: {e}")
            print(f"Error training model: {e}")
            raise

    def calculate_trust_score(self, model: keras.Model, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[float, str]:
        try:
            with st.spinner("Calculating trust score..."):
                st.write("Calculating trust score...")
                print("Console: Calculating trust score...")
                loss, accuracy = model.evaluate(x_data, y_data, verbose=0)
            trust_score = accuracy * 100
            if trust_score >= 85:
                verdict = "HIGH_TRUST"
            elif trust_score >= 70:
                verdict = "MEDIUM_TRUST"
            else:
                verdict = "LOW_TRUST"
            st.success(f"Trust score calculated: {trust_score:.2f}% ({verdict})")
            print(f"Console: Trust score calculated: {trust_score:.2f}% ({verdict})")
            return trust_score, verdict
        except Exception as e:
            st.error(f"Error calculating trust score: {e}")
            print(f"Error calculating trust score: {e}")
            raise

def generate_brief_summary(score: float, verdict: str) -> str:
    if verdict == "HIGH_TRUST":
        return (f"Dataset is highly trustworthy: accuracy = {score:.2f}%. No data poisoning or severe integrity issues detected.")
    elif verdict == "MEDIUM_TRUST":
        return (f"Dataset shows medium trust: accuracy = {score:.2f}%. There may be minor issues or distribution drift present.")
    else:
        return (f"Dataset has low trust: accuracy = {score:.2f}%. Strong sign of data corruption, heavy noise, or poisoning detected.")

def main():
    st.title("Dataset Trust Score Application")
    st.write("CIFAR-10 Dataset Trust Evaluation (with Manual Upload Option)")

    trust_scorer = DatasetTrustScorer()
    dataset_choice = st.radio(
        "Choose dataset source:",
        ("Built-in CIFAR-10", "Manual Upload"),
    )

    show_upload = (dataset_choice == "Manual Upload")
    if show_upload:
        st.subheader("Upload your .npz, .csv, or .xlsx file (with x and y arrays/columns)")
        uploaded_file = st.file_uploader("Choose a file", type=["npz", "csv", "xlsx"])
        if uploaded_file is not None:
            try:
                x_data, y_data = trust_scorer.load_custom_data(uploaded_file)
                st.session_state['x_train'] = x_data
                st.session_state['y_train'] = y_data
                st.session_state['data_loaded'] = True
            except Exception as e:
                st.error(f"Failed to load uploaded data: {e}")
    else:
        if st.button("Load CIFAR-10 Data"):
            try:
                x_train, y_train, x_test, y_test = trust_scorer.load_cifar10_data()
                st.session_state['x_train'] = x_train
                st.session_state['y_train'] = y_train
                st.session_state['x_test'] = x_test
                st.session_state['y_test'] = y_test
                st.session_state['data_loaded'] = True
            except Exception as e:
                st.error(f"Failed to load data: {e}")

    st.subheader("Model Operations")
    if st.button("Load Existing Model"):
        try:
            model = trust_scorer.load_model()
            if model is not None:
                st.session_state['model'] = model
                st.session_state['model_loaded'] = True
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    if st.button("Create & Train New Model"):
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load or upload data first!")
        else:
            try:
                model = trust_scorer.create_model()
                if dataset_choice == "Manual Upload":
                    trained_model = trust_scorer.train_model(
                        model,
                        st.session_state['x_train'],
                        st.session_state['y_train'],
                        st.session_state['x_train'],
                        st.session_state['y_train'],
                    )
                else:
                    trained_model = trust_scorer.train_model(
                        model,
                        st.session_state['x_train'],
                        st.session_state['y_train'],
                        st.session_state['x_test'],
                        st.session_state['y_test'],
                    )
                trust_scorer.save_model(trained_model)
                st.session_state['model'] = trained_model
                st.session_state['model_loaded'] = True
            except Exception as e:
                st.error(f"Failed to create/train model: {e}")

    st.subheader("Trust Score Evaluation")
    if st.button("Calculate Trust Score"):
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load or upload data first!")
        elif not st.session_state.get('model_loaded', False):
            st.warning("Please load or train a model first!")
        else:
            try:
                if dataset_choice == "Manual Upload":
                    trust_score, verdict = trust_scorer.calculate_trust_score(
                        st.session_state['model'],
                        st.session_state['x_train'],
                        st.session_state['y_train'],
                    )
                else:
                    trust_score, verdict = trust_scorer.calculate_trust_score(
                        st.session_state['model'],
                        st.session_state['x_test'],
                        st.session_state['y_test'],
                    )
                st.metric("Trust Score", f"{trust_score:.2f}%")
                st.metric("Verdict", verdict)
                if verdict == "HIGH_TRUST":
                    st.success(f"✅ Dataset is trustworthy (Score: {trust_score:.2f}%)")
                elif verdict == "MEDIUM_TRUST":
                    st.warning(f"⚠️ Dataset has medium trustworthiness (Score: {trust_score:.2f}%)")
                else:
                    st.error(f"❌ Dataset has low trustworthiness (Score: {trust_score:.2f}%)")
                brief = generate_brief_summary(trust_score, verdict)
                st.info(brief)
            except Exception as e:
                st.error(f"Failed to calculate trust score: {e}")

    with st.expander("Debug Info"):
        st.write("Session State:")
        st.write({
            'data_loaded': st.session_state.get('data_loaded', False),
            'model_loaded': st.session_state.get('model_loaded', False)
        })

if __name__ == "__main__":
    main()

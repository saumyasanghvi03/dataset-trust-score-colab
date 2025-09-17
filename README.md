# dataset-trust-score-colab

Google Colab notebook and scripts for Dataset Trust Score System using CIFAR-10. Contains notebook and helper files as per project plan.

## Files

- **app.py** - Main standalone application for Dataset Trust Score analysis
- **cifar10_sample.npz** - Small demonstration subset of CIFAR-10 data (100 samples)
- **README.md** - This documentation file

## CIFAR-10 Sample Dataset

### ⚠️ Important Notice

The `cifar10_sample.npz` file contains a **small, shareable sample** of CIFAR-10 data (100 samples) for demonstration purposes only. This file is provided for:

- Quick testing without large downloads
- Educational demonstrations
- Sharing small examples

### 🚨 For Actual Experiments

**DO NOT use this sample file for real experiments or research.** For actual Dataset Trust Score analysis, always use the full CIFAR-10 dataset by allowing Keras to automatically download it:

```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

The full dataset contains 50,000 training samples and 10,000 test samples, which are necessary for meaningful trust score calculations.

## Usage

1. Run the main application:
   ```bash
   python app.py --dataset-type clean
   python app.py --dataset-type poisoned --poison-rate 0.1
   ```

2. The application will automatically download the full CIFAR-10 dataset when needed.

3. For demonstration with the sample file, you can modify the code to load from `cifar10_sample.npz`, but remember this is only for testing purposes.

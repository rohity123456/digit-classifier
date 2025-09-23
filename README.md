# Digit Classifier

A simple neural network-based digit classifier built with Python. This project demonstrates how to train and evaluate a neural network to recognize handwritten digits using popular machine learning libraries.

It includes a Flask app for serving predictions via an API for any handwritten digit image.

## Features

- Neural network implementation for digit classification
- Training and evaluation on standard datasets (e.g., MNIST)
- Easy-to-understand code structure
- Customizable model architecture and hyperparameters
- Flask API for serving predictions for handwritten digit images

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/yourusername/digit-classifier.git
  cd digit-classifier
  ```
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

Train the model:
```bash
python train.py
```

Evaluate the model:
```bash
python evaluate.py
```

Prediction API (Flask app):
  ```bash
  python app/app.py
  ```
  endpoint: `POST /predict` with image file in form-data.


## Project Structure

```
digit-classifier/
src/
  ├── models/             # Forward pass implementations
  ├── training/            # Training script
  ├── inference/          # Inference utilities
  ├── app/              # Flask app for serving predictions
  └── README.md           # Project documentation
```

## Dataset

This project uses the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset of handwritten digits. The dataset is automatically downloaded when running the training script.

## Customization

- Modify `models/` to experiment with different neural network architectures.
- Adjust hyperparameters in `train.py` for better performance.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License.

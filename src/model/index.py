"""
src/model.py

Forward-pass module for educational MNIST MLP (no training).
Provides:
 - simple weight initialization (He for ReLU)
 - forward(X, params) that returns softmax probabilities and a cache
 - activation helpers (relu, softmax)
 - small test runner to verify shapes and run a single forward pass

Usage (from terminal):
  python src/model.py

This script will try to load MNIST (28x28) using tensorflow.keras.datasets if
available, otherwise will fallback to sklearn's fetch_openml (requires internet).

Keep this file minimal and educational — we'll add training code in later steps.
"""

from typing import Tuple, Dict
import numpy as np

# ----------------------- Utilities / activations -----------------------

def relu(x: np.ndarray) -> np.ndarray:
    """Elementwise ReLU activation."""
    return np.maximum(0, x)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax with numerical stability.

    logits: shape (batch, classes)
    returns: probs shape (batch, classes)
    """
    z = logits - np.max(logits, axis=1, keepdims=True)
    expz = np.exp(z)
    probs = expz / np.sum(expz, axis=1, keepdims=True)
    return probs


# ----------------------- Initialization helpers -----------------------

def he_init(fan_in: int, fan_out: int, seed: int | None = None) -> np.ndarray:
    """He (Kaiming) initialization for ReLU: N(0, sqrt(2/fan_in))."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)


def zeros(shape: Tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=float)


# ----------------------- Forward pass (vectorized) -----------------------

def forward(X: np.ndarray, params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict]:
    """Compute forward pass for a 2-layer MLP.

    Args:
        X: input batch, shape (batch, D)
        params: dictionary with keys 'W1','b1','W2','b2'

    Returns:
        probs: softmax probabilities, shape (batch, C)
        cache: dictionary with intermediate arrays for later use (z1,a1,z2)
    """
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

    # Sanity checks on shapes
    assert X.ndim == 2, "X must be 2D (batch, features)"
    batch, D = X.shape
    assert W1.shape[0] == D, f"W1 first dim should match input D ({D}), got {W1.shape}"
    H = W1.shape[1]
    C = W2.shape[1]

    # Layer 1 (input -> hidden)
    z1 = X.dot(W1) + b1  # shape (batch, H) ; b1 broadcast if shape (1,H)
    a1 = relu(z1)        # shape (batch, H)

    # Layer 2 (hidden -> logits)
    z2 = a1.dot(W2) + b2  # shape (batch, C)
    probs = softmax(z2)   # shape (batch, C)

    cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'probs': probs}
    return probs, cache


# ----------------------- MNIST loader helper -----------------------

def load_mnist_subset(n_train: int = 1000, n_val: int = 200, seed: int | None = 42):
    """Try to load MNIST (28x28) and return small train/val/test splits.

    Preference order:
      1) tensorflow.keras.datasets.mnist (if tensorflow is installed)
      2) sklearn.datasets.fetch_openml('mnist_784') (requires internet)

    Returns:
      X_train, y_train, X_val, y_val, X_test, y_test (numpy arrays)
    """
    try:
        from sklearn.datasets import fetch_openml
        mn = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        print("Loaded MNIST via sklearn.datasets.fetch_openml")
        print("mn data shape:", mn.data.shape)
        print("mn target shape:", mn.target.shape)
        X = mn.data.reshape(-1, 28, 28)
        y = mn.target.astype(int)
    except Exception as e:
        raise RuntimeError("Could not load MNIST.") from e

    # normalize to [0,1]
    X = X.astype(np.float32) / 255.0

    # flatten to (N, 784)
    N = X.shape[0]
    X = X.reshape(N, -1)

    # simple deterministic shuffle
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    X = X[perm]
    y = y[perm]

    # split
    n_train = min(n_train, N - n_val - 100)
    n_val = min(n_val, N - n_train - 50)
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val: n_train+n_val+200]
    y_test = y[n_train+n_val: n_train+n_val+200]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ----------------------- Small test runner -----------------------
if __name__ == "__main__":
    import sys

    print("Running forward-pass sanity test for MNIST MLP (no training)\n")

    # hyperparameters
    input_dim = 28 * 28
    hidden_dim = 128
    output_dim = 10

    # initialize parameters
    print("Initializing parameters (He init for W1,W2)")
    W1 = he_init(input_dim, hidden_dim, seed=123)
    b1 = zeros((1, hidden_dim))
    W2 = he_init(hidden_dim, output_dim, seed=456)
    b2 = zeros((1, output_dim))
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    # load small subset of MNIST
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_subset(n_train=512, n_val=128)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    print("Data shapes:")
    print(" X_train", X_train.shape)
    print(" X_val  ", X_val.shape)
    print(" X_test ", X_test.shape)

    # take a small batch
    X_batch = X_train[:8]
    print("\nRunning forward pass on a small batch of 8 samples...")
    probs, cache = forward(X_batch, params)

    print("Shapes after forward pass:")
    print(" probs", probs.shape)
    print(" z1", cache['z1'].shape)
    print(" a1", cache['a1'].shape)
    print(" z2", cache['z2'].shape)

    print("\nExample probabilities (first two samples):")
    with np.printoptions(precision=4, suppress=True):
        print(probs[:2])

    print("\nSum of probabilities per sample (should be 1):", np.sum(probs, axis=1))
    print("Done.")

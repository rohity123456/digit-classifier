"""
src/gradients.py

Backward-pass implementation (gradients) for educational MNIST MLP.

This module provides:
 - one_hot(y, num_classes)
 - relu_derivative(z)
 - backward(cache, params, Y) that computes gradients for W1,b1,W2,b2
 - grad_check(...) to numerically verify gradients

Usage (as script):
  python src/gradients.py

It imports `forward` from `src/model.py` (module must be importable from project root).
"""

from typing import Dict, Tuple
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# ...existing code...
# ----------------------- Helper: one-hot encoding -----------------------

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels (shape (batch,)) to one-hot (batch, num_classes)."""
    y = np.asarray(y, dtype=int)
    batch = y.shape[0]
    Y = np.zeros((batch, num_classes), dtype=np.float32)
    Y[np.arange(batch), y] = 1.0
    return Y


# ----------------------- Activation derivative -----------------------

def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where z>0 else 0.

    z: pre-activation array (batch, H) or (H,)
    returns: same shape, 1.0 where z>0, else 0.0
    """
    return (z > 0).astype(np.float32)


# ----------------------- Backward pass -----------------------

def backward(cache: Dict[str, np.ndarray], params: Dict[str, np.ndarray], Y: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute gradients for a 2-layer MLP using cached forward values.

    Args:
        cache: dictionary returned by forward() containing X, z1, a1, z2, probs
        params: dictionary with 'W1','b1','W2','b2' (used for shapes)
        Y: integer labels shape (batch,) or one-hot shape (batch, C)

    Returns:
        grads: dictionary with keys 'dW1','db1','dW2','db2'
    """
    # Unpack cache
    X = cache['X']        # (batch, D)
    z1 = cache['z1']      # (batch, H)
    a1 = cache['a1']      # (batch, H)
    probs = cache['probs']# (batch, C)

    W1 = params['W1']
    W2 = params['W2']

    m, C = probs.shape

    # Convert Y to one-hot if necessary
    if Y.ndim == 1:
        Y_oh = one_hot(Y, C)
    else:
        Y_oh = Y

    # dZ2: derivative of loss wrt logits z2 (softmax + cross-entropy)
    # For mean loss over batch: dZ2 = (probs - Y) / m
    dZ2 = (probs - Y_oh) / m            # (m, C)

    # Gradients for W2 and b2
    dW2 = a1.T.dot(dZ2)                 # (H, C)
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1, C)

    # Backprop into hidden layer
    dA1 = dZ2.dot(W2.T)                 # (m, H)
    dZ1 = dA1 * relu_derivative(z1)     # (m, H)

    # Gradients for W1 and b1
    dW1 = X.T.dot(dZ1)                  # (D, H)
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1, H)

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads


# ----------------------- Numerical gradient check -----------------------

def grad_check(X_small: np.ndarray, Y_small_oh: np.ndarray, params: Dict[str, np.ndarray], epsilon: float = 1e-5):
    """Numerical gradient check for a few selected weights to validate backward().

    Computes numeric gradients using finite differences and compares to analytic grads.
    Prints differences for inspection.
    """
    from src.model.index import forward  # local import to avoid circular issues at module import

    probs, cache = forward(X_small, params)
    analytic = backward(cache, params, Y_small_oh)

    def loss_for_params(W1p, b1p, W2p, b2p):
        p, _ = forward(X_small, {'W1': W1p, 'b1': b1p, 'W2': W2p, 'b2': b2p})
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(np.sum(Y_small_oh * np.log(p), axis=1))

    dW1_a, db1_a, dW2_a, db2_a = analytic['dW1'], analytic['db1'], analytic['dW2'], analytic['db2']

    # pick a few indices to check (ensure indices are valid for shapes)
    idxs_w1 = [(0,0), (min(1, dW1_a.shape[0]-1), min(2, dW1_a.shape[1]-1)), (min(2, dW1_a.shape[0]-1), min(3, dW1_a.shape[1]-1))]
    idxs_w2 = [(0,0), (min(1, dW2_a.shape[0]-1), min(2, dW2_a.shape[1]-1)), (min(2, dW2_a.shape[0]-1), min(3, dW2_a.shape[1]-1))]

    print("\nGradient check (finite differences vs analytic):")
    for (i,j) in idxs_w1:
        W1p = params['W1'].copy(); W1p[i,j] += epsilon
        loss_plus = loss_for_params(W1p, params['b1'], params['W2'], params['b2'])
        W1m = params['W1'].copy(); W1m[i,j] -= epsilon
        loss_minus = loss_for_params(W1m, params['b1'], params['W2'], params['b2'])
        num = (loss_plus - loss_minus) / (2*epsilon)
        print(f"W1[{i},{j}]: analytic={dW1_a[i,j]:.8f}, numeric={num:.8f}, diff={abs(dW1_a[i,j]-num):.2e}")

    for (i,j) in idxs_w2:
        W2p = params['W2'].copy(); W2p[i,j] += epsilon
        loss_plus = loss_for_params(params['W1'], params['b1'], W2p, params['b2'])
        W2m = params['W2'].copy(); W2m[i,j] -= epsilon
        loss_minus = loss_for_params(params['W1'], params['b1'], W2m, params['b2'])
        num = (loss_plus - loss_minus) / (2*epsilon)
        print(f"W2[{i},{j}]: analytic={dW2_a[i,j]:.8f}, numeric={num:.8f}, diff={abs(dW2_a[i,j]-num):.2e}")


# ----------------------- Self-test when run as script -----------------------
if __name__ == "__main__":
    def main():       
      from src.model.index import forward, he_init, zeros, load_mnist_subset

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
      Y_batch = y_train[:8]
      print("\nRunning forward pass on a small batch of 8 samples...")
      probs, cache = forward(X_batch, params)
      m, C = probs.shape
      grads = backward(cache, params, Y_batch)

      print("Shapes of gradients:")
      for k,v in grads.items():
          print(k, v.shape)

      print("\nExample gradient values (dW1 first row):")
      with np.printoptions(precision=4, suppress=True):
          print(grads['dW1'][0])

      # Perform gradient check on the tiny example
      Y_small_oh = one_hot(Y_batch, C)
      grad_check(X_batch, Y_small_oh, params)

      print("Done.")
    main()
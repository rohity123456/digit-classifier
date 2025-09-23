"""
src/train.py

Full training script for the educational 2-layer MLP on MNIST (NumPy implementation).

Features:
 - Loads MNIST via src.model.load_mnist_subset
 - Initializes parameters (He init)
 - Trains with mini-batch SGD, shuffling each epoch
 - Tracks & prints train/validation loss and accuracy per epoch
 - Evaluates on test set after training
 - Saves trained parameters to `saved_params.npz`

Run from project root:
    python src/train.py

This script depends on src/model.py and src/gradients.py being present and importable.
"""

import time
from typing import Dict
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model.index import forward, he_init, zeros, load_mnist_subset
from src.gradients.index import backward, one_hot
import matplotlib.pyplot as plt

CHECKPOINT_PATH = "saved_params.npz"

# ----------------------- Loss & metrics -----------------------

def cross_entropy_loss(probs: np.ndarray, Y_onehot: np.ndarray) -> float:
    eps = 1e-12
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(np.sum(Y_onehot * np.log(probs), axis=1))


def accuracy_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == labels)


# ----------------------- Training routine -----------------------

def save_checkpoint(path: str, params: dict, epoch: int):
  """Save parameters and metadata to a .npz file atomically.


  We write to a temporary file then rename to avoid partial files.
  """
  tmp_path = path + ".tmp"
  # Prepare arrays to save
  np.savez(tmp_path,
  W1=params['W1'], b1=params['b1'], W2=params['W2'], b2=params['b2'],
  epoch=np.array(epoch, dtype=np.int32))
  # atomic replace
  try:
    os.replace(tmp_path + '.npy', tmp_path + '.npy')
  except Exception:
  # np.savez creates .npz directly; just rename/move
    pass
  # move tmp to final (np.savez created a .npz file at tmp_path)
  final_tmp = tmp_path + ".npz"
  if os.path.exists(final_tmp):
    os.replace(final_tmp, path)
  else:
  # if np.savez wrote directly to tmp_path without .npz suffix
    if os.path.exists(tmp_path):
      os.replace(tmp_path, path)




def load_checkpoint(path: str):
  """Load parameters and metadata from a checkpoint. Returns (params, epoch) or (None, 0)."""
  if not os.path.exists(path):
    return None
  data = np.load(path)
  params : Dict[str, np.ndarray] = {'W1': data['W1'], 'b1': data['b1'], 'W2': data['W2'], 'b2': data['b2']}
  return params

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int = 28*28,
    hidden_dim: int = 128,
    output_dim: int = 10,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.1,
    seed: int = 123,
):
    rng = np.random.RandomState(seed)
    from typing import Optional
    loaded_params: Optional[Dict[str, np.ndarray]] = load_checkpoint("saved_params.npz")
    if loaded_params:
      print(f"Resuming training from checkpoint {CHECKPOINT_PATH}")
      print(f"First weights: {loaded_params['W1'].ravel()[:5]}")
      params: Dict[str, np.ndarray] = loaded_params
    else:
      # initialize parameters
      W1 = he_init(input_dim, hidden_dim, seed=seed).astype(np.float32)
      b1 = zeros((1, hidden_dim)).astype(np.float32)
      W2 = he_init(hidden_dim, output_dim, seed=seed+1).astype(np.float32)
      b2 = zeros((1, output_dim)).astype(np.float32)
      params: Dict[str, np.ndarray] = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    n_train = X_train.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"Starting training: epochs={epochs}, batch_size={batch_size}, steps_per_epoch={steps_per_epoch}")
    start_time = time.time()

    for ep in range(1, epochs + 1):
        # shuffle training data
        perm = rng.permutation(n_train)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            X_batch = X_shuf[i:i+batch_size]
            y_batch = y_shuf[i:i+batch_size]
            m = X_batch.shape[0]

            # forward
            probs, cache = forward(X_batch, params)
            Y_batch_oh = one_hot(y_batch, output_dim)
            loss = cross_entropy_loss(probs, Y_batch_oh)
            epoch_loss += loss * m

            # backward
            grads = backward(cache, params, Y_batch_oh)

            # update (SGD)
            params['W1'] -= lr * grads['dW1']
            params['b1'] -= lr * grads['db1']
            params['W2'] -= lr * grads['dW2']
            params['b2'] -= lr * grads['db2']

        # end epoch
        epoch_loss /= n_train

        # evaluate train and val (full)
        train_probs, _ = forward(X_train, params)
        val_probs, _ = forward(X_val, params)
        train_loss = cross_entropy_loss(train_probs, one_hot(y_train, output_dim))
        val_loss = cross_entropy_loss(val_probs, one_hot(y_val, output_dim))
        train_acc = accuracy_from_probs(train_probs, y_train)
        val_acc = accuracy_from_probs(val_probs, y_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - start_time
        print(f"Epoch {ep:3d}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f} | elapsed {elapsed:.1f}s")

    # final test evaluation
    test_probs, _ = forward(X_test, params)
    test_acc = accuracy_from_probs(test_probs, y_test)
    test_loss = cross_entropy_loss(test_probs, one_hot(y_test, output_dim))

    print(f"\nTraining finished in {time.time() - start_time:.1f}s. Test loss {test_loss:.4f}, Test acc {test_acc:.4f}")

    # save parameters
    np.savez(CHECKPOINT_PATH, W1=params['W1'], b1=params['b1'], W2=params['W2'], b2=params['b2'])
    print(f'Saved parameters to {CHECKPOINT_PATH}')

    # simple plots
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss curves')
    plt.savefig('loss_curve.png')

    plt.figure()
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy curves')
    plt.savefig('acc_curve.png')

    return params, history


if __name__ == '__main__':
    # load data (small subset for quick runs; change n_train param to use full dataset)
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_subset(n_train=10000, n_val=2000)
    
    params, history = train(
        X_train, y_train, X_val, y_val, X_test, y_test,
        input_dim=28*28,
        hidden_dim=128,
        output_dim=10,
        epochs=20,
        batch_size=128,
        lr=0.1,
        seed=123,
    )

    print('Done.')

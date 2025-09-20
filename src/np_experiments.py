import numpy as np

batch = 8
a = np.arange(batch)
print("a:", a)
eps = 1e-12
p = np.zeros((batch, 10))
p = np.clip(p, eps, 1 - eps)
print("p:", p)
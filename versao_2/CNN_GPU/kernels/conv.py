import numpy as np

w = np.array([[[[-0.03, 0.11], [0.06, 0.08]]], [[[0.05, -0.14], [-0.04, 0.18]]]])

a = np.array([[0.83, 0.38, -0.30], [0.99, -0.80, 0.44], [0.71, 0.48, 0.53]])

w1 = w[0, :, :, :]
w2 = w[1, :, :, :]
print(a.shape)

z = np.array([[[[np.sum(w1 * a[:1, :1]), np.sum(w1 * a[:1, 1:])], [np.sum(w1 * a[1:, :1]), np.sum(w1 * a[1:, 1:])]]],\
 [[[np.sum(w2 * a[:1, :1]), np.sum(w2 * a[:1, 1:])], [np.sum(w2 * a[1:, :1]), np.sum(w2 * a[1:, 1:])]]]])
print(z.shape)
print(z)


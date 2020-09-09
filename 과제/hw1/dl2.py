import numpy as np
import matplotlib.pyplot as plt
import math
np.random.seed(1337)
Z = np.random.normal(0, 1, (2, 10000))
sigma = np.array([[math.sqrt(1.9/2), math.sqrt(0.1/2)],
[math.sqrt(1.9/2), -math.sqrt(0.1/2)]])
X = np.matmul(sigma, Z)
mean = [0.0, 0.0]
cov = [[1.0, 0.9],[0.9, 1.0]]
MN = np.random.multivariate_normal(mean, cov, 10000).T
plt.subplot(1, 3, 1)
plt.scatter(Z[0], Z[1], s=0.1)
plt.subplot(1, 3, 2)
plt.scatter(X[0], X[1], s=0.1)
plt.subplot(1, 3, 3)
plt.scatter(MN[0], MN[1], s=0.1)

plt.show()
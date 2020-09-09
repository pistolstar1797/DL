import numpy as np
import matplotlib.pyplot as plt
samples = np.random.exponential(1, 1000000)
y, x, p = plt.hist(samples, bins=500, density=True)
fx = np.linspace(min(x), max(x), 500)
fy = np.exp(-fx)
plt.plot(fx, fy) 
plt.show()
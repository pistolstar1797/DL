import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)

n = 1000
d = 100

X = np.vstack([np.random.normal(0.1, 1, (n//2, d)),
               np.random.normal(-0.1, 1, (n//2, d))])
y = np.hstack([np.ones(n//2), -1*np.ones(n//2)])
w0 = np.random.normal(0, 1, d)
w = w0

def subGrad(w):
    SG = np.zeros(d)
    SG += 0.1 * w
    for i in range(n):
        Xi = X[i, :]
        yi = y[i]
        yiwTXi = yi * np.matmul(w.T, Xi)
        if 1 > yiwTXi:
            SG += -(yi/n)*Xi
    return SG

def lossVal(w):
    ret = 0.0
    ret += (0.1/2)*np.matmul(w.T, w)
    for i in range(n):
        Xi = X[i, :]
        yi = y[i]
        yiwTXi = yi * np.matmul(w.T, Xi)
        ret += np.max([0.0, 1 - yiwTXi])/n
    return ret

def accuracy(w):
    match = 0
    for i in range(n):
        Xi = X[i, :]
        yi = y[i]
        wTXi = np.matmul(w.T, Xi)
        if np.sign(wTXi) == yi:
            match += 1
    return match/n

iter = []
f_value = []
c_accur = []

for i in range(100):
    w -= subGrad(w)*0.01
    iter.append(i+1)
    f_value.append(np.log(lossVal(w)))
    c_accur.append(accuracy(w))

plt.subplot(1, 2, 1)
plt.plot(iter, f_value)
plt.subplot(1, 2, 2)
plt.plot(iter, c_accur)
plt.show()
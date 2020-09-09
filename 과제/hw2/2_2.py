import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
np.random.seed(1337)

n = 5
A = np.random.normal(0, 1, (n, n))
S = A.dot(A.T)

def projection(X):
    eValue, eVector = np.linalg.eig(X)
    index = eValue.argsort()[::-1]
    eValue = eValue[index]
    eVector = eVector[:,index]
    for i in range(n):
        eValue[i] = np.max([0.0, eValue[i]])
    return eVector.dot(np.diag(eValue)).dot(eVector.T)
    
def target(X):
    ST = S.T
    tr = np.trace(ST.dot(X))
    det_log = np.log(np.linalg.det(X) + 1e-10)
    abs = np.sum(np.absolute(X))
    return tr - det_log + 0.1 * abs

iter = []
f_value = []

temp = np.random.normal(0, 1, (n, n))
X = temp.dot(temp.T)

for i in range(500):
    X -= grad(target)(X)*0.01
    X = projection(X)
    iter.append(i+1)
    f_value.append(target(X))

plt.plot(iter, f_value)
plt.show()
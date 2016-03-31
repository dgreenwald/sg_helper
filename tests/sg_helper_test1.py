"""1-D SGHelper Example File"""

import matplotlib.pyplot as plt
import numpy as np
from sg_helper import SGHelper

def f(x): return (1.0 / ((np.abs(0.5 - (x ** 4))) + 0.1))

sgh = SGHelper(fcn=f, dim=1, init_level=2, tol=1e-1, maxit=100, grid_type='ModLinear')

N = 10000
x_dense = np.linspace(0.0, 1.0, N)
y_dense = f(x_dense)
y_hat = sgh.evaluate(x_dense)

plt.figure()
plt.plot(x_dense, y_dense, color='blue', label='True')
plt.plot(x_dense, y_hat, color='red', label='Approx')
plt.legend(loc='upper left')
plt.savefig('test1.pdf')
plt.close(fig)

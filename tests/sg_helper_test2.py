""" 2-D SGHelper Example File """

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sg_helper import SGHelper

def f(x): 
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    return (1.0 / (np.abs(0.5 - np.sum((x ** 4), axis=0)) + 0.1))

# Set bounds
lb = None
ub = None
# lb = np.array((-0.5, 0.2))
# ub = np.array((1.5, 0.8))

sgh = SGHelper(fcn=f, dim=2, init_level=2, tol=0.1, maxit=100, grid_type='ModLinear',
               lb=lb, ub=ub)

points = sgh.grid_points()
vals = f(points)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(60, 270)
ax.scatter(points[0, :], points[1, :], vals, s=1)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.savefig('test_2d.pdf')
plt.close()
""" SGHelper: a numpy interface for the SG++ library """

import numpy as np
from pysgpp import DataVector, Grid, createOperationHierarchisation, \
    SurplusRefinementFunctor, createOperationEval

class SGHelper:
    """ Wrapper object for SG++ """
    
    def __init__(self, fcn, dim, init_level=1, tol=1.0, maxit=100,
                 max_pts=10000, grid_type='ModLinear', lb=None, ub=None):
        """ Constructor that computes and refines grid. Arguments are:

        fcn (function): function to approximate, takes a numpy column vector as an argument 
            and returns a float.

        dim (int): the dimension of the problem (e.g., the length of the vector input to fcn).

        tol (float): the maximum allowed hierarchical surplus.

        maxit (int): the maximum allowed number of refinements.

        max_pts (int): the maximum allowed number of gridpoints.

        grid_type (string): the type of hierarchical basis to construct. This can be any of the types 
            supported by SG++, but may include e.g.:

            'Linear': linear basis functions equal to zero at the boundaries.
            'ModLinear': linear basis functions that extrapolate at the boundaries.
            'LinearBoundary': linear basis functions that explicitly include the boundaries.

            The same naming system holds for other types of basis functions, e.g., 'ModBSpline'.

        lb (numpy vector with length dim): the lower bound of the cube on which to evaluate the functions, 
            defaults to zero.

        ub (numpy vector with length dim): the upper bound of the cube on which to evaluate the functions, 
            defaults to one. """

        # Load members
        self.fcn = fcn
        self.dim = dim
        self.init_level = init_level
        self.tol = tol
        self.maxit = maxit
        self.max_pts = max_pts
        self.grid_type = grid_type
        self.lb = lb
        self.ub = ub

        if self.lb is None:
            self.lb = np.zeros(self.dim)
        if self.ub is None:
            self.ub = np.ones(self.dim)
        
        # Initialize grid
        attr_name = 'create' + grid_type + 'Grid'
        self.grid = getattr(Grid, attr_name)(self.dim)
        self.gridStorage = self.grid.getStorage()
        self.gridGen = self.grid.createGridGenerator()
        self.gridGen.regular(self.init_level)

        # Initialize alpha
        self.alpha = DataVector(self.gridStorage.size())

        # Construct adaptive sparse grid
        done = False
        it = 0
        while not done:
            it += 1

            scaled_0_1_point = np.zeros(self.dim)

            # set function values in self.alpha
            for i_pt in xrange(self.gridStorage.size()):
                gp = self.gridStorage.get(i_pt)
                for i_dim in xrange(self.dim):
                    scaled_0_1_point[i_dim] = gp.getCoord(i_dim)
                point = self.scale_from_0_1(scaled_0_1_point)
                self.alpha[i_pt] = self.fcn(point)[0]

            createOperationHierarchisation(self.grid).doHierarchisation(self.alpha)    

            if it >= maxit or len(self.alpha) >= max_pts:
                done = True
            else:
                self.gridGen.refine(SurplusRefinementFunctor(self.alpha, len(self.alpha), self.tol))
                # print "refinement step {}, new grid size: {}".format(refnum+1, self.gridStorage.size())

                if len(self.alpha) == self.gridStorage.size():
                    done = True
                else:
                    # # extend self.alpha vector (new entries uninitialized)
                    self.alpha.resize(self.gridStorage.size())

    def evaluate(self, points):
        """ Evaluate function on numpy array. Each column should be a single point. """

        if len(points.shape) == 1:
            points = points[np.newaxis, :]

        opEval = createOperationEval(self.grid) 
        vec = DataVector(self.dim)
        vals = np.zeros(points.shape[1])

        for i_pt in xrange(points.shape[1]):
            scaled_0_1_point = self.scale_to_0_1(points[:, i_pt])
            for i_dim in xrange(points.shape[0]):
                vec.set(i_dim, scaled_0_1_point[i_dim])
            vals[i_pt] = opEval.eval(self.alpha, vec)

        return vals

    def size(self):
        """ Return size of current grid """

        return self.gridStorage.size()

    def scale_to_0_1(self, point):
        return((point - self.lb) / (self.ub - self.lb))

    def scale_from_0_1(self, scaled_0_1_point):
        return ((self.ub - self.lb) * scaled_0_1_point + self.lb)

    def grid_points(self):
        """ Return current grid points as numpy array. Each column is one point. """

        points = np.zeros((self.dim, self.size()))
        scaled_0_1_point = np.zeros(self.dim)
        for i_pt in xrange(self.size()):
            gp = self.gridStorage.get(i_pt)
            for i_dim in xrange(self.dim):
                scaled_0_1_point[i_dim] = gp.getCoord(i_dim)
            points[:, i_pt] = self.scale_from_0_1(scaled_0_1_point)

        return points

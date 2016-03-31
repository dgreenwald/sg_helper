SGHelper: a numpy interface for the SG++ adaptive sparse grids library

Requires the SG++ library of Dirk Pflueger, which can be found at http://sgpp.sparsegrids.org/.

This module provides a class SGHelper for easy interface with SG++ capabilities using numpy arrays.
Functions can be approximated on arbitrary cubes by providing lower and upper bounds for each argument.
The constructor for the class automatically creates a grid that is refined to a set tolerance.
The grid can then be used to evaluate the function on a numpy array of points. Each column should be one input point.

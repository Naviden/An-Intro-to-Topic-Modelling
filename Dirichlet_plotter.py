
# CODE SOURCE :  http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from functools import reduce
from math import gamma
from operator import mul
import math


corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])


# Mid-points of triangle sides opposite of each corner
midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) /
             2.0 for i in range(3)]


def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    s = [(corners[i] - midpoints[i]).dot(xy -
                                         midpoints[i]) / 0.75 for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)


class Dirichlet(object):
    def __init__(self, alpha):

        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
            reduce(mul, [gamma(a) for a in self._alpha])

    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])


def draw_pdf_contours(dist, labels, nlevels=15, subdiv=8, **kwargs):

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    plt.text(corners[0][0]-0.3, corners[0][1], labels[0].rjust(10))
    plt.text(corners[1][0] + 0.03, corners[1][1], labels[1].ljust(10))
    plt.text(corners[2][0] - 0.14, corners[2][1] + 0.03, labels[2].center(10))

# draw_pdf_contours(Dirichlet([0.999, 5, 5]), ['AAAA', 'BBBBBBBBBBBB', 'CCCCCC'])

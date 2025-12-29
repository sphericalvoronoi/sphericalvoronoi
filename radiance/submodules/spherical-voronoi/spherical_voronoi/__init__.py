from . import _C

def spherical_voronoi(sites, directions, tau, colors):
    return _C.forward(sites, directions, tau, colors)
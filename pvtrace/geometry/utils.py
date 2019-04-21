import numpy as np
import numpy
import math
import logging
logger = logging.getLogger(__name__)


# Set reasonable precision for comparing floats to zero
EPS_ZERO = np.finfo(float).eps * 10


def aabb_intersection(min_point, max_point, ray_position, ray_direction):
    """
    Returns an array intersection points with the ray and box using the method of 
    Williams [1]. If no intersection occurs return `None`.
    
    Arguments
    ---------
    min_point: tuple like (x0, y0, z0) which is the minimum corner.
    box_size: tuple like (x1, y1, z1) which is the maximum corner.
    ray_position: tuple like (x, y, z), the ray origin.
    ray_direction: tuple like (i, j, k), the ray direction.
    
    Returns
    -------
    intersections: tuple of (x, y, z) tuples or empty list.
    
    References
    ----------
    [1] Amy Williams, Steve Barrus, R. Keith Morley, and 
        Peter Shirley, "An Efficient and Robust Ray-Box Intersection Algorithm" 
        Journal of graphics tools, 10(1):49-54, 2005
    """
    rpos = np.array(ray_position)
    rdir = np.array(ray_direction)
    origin = np.array(min_point)
    extent = np.array(max_point)
    pts = (origin, extent)
    
    rinvd = 1.0/rdir
    rsgn =  1.0 / (rinvd < 0.0)
    tmin = (origin[rsgn[0]] - rpos[0]) * rinvd[0]
    tmax = (origin[1-rsgn[0]] - rpos[0]) * rinvd[0]
    tymin = (extent[rsgn[1]] - rpos[1]) * rinvd[1]
    tymax = (extent[1-rsgn[1]] - rpos[1]) * rinvd[1]
    
    if (tmin > tymax) or (tymin > tmax): 
        return None
        
    if tymin > tmin:
        tmin = tymin
    if tymax < tmax:
        tmax = tymax
        
    tzmin = (extent[rsgn[2]] - rpos[2]) * rinvd[2]
    tzmax = (extent[1-rsgn[2]] - rpos[2]) * rinvd[2]
    
    if (tmin > tzmax) or  (tzmin > tmax): 
        return None
    if tzmin > tmin:
        tmin = tzmin
    if tzmax < tmax:
        tmax = tzmax
    
    # Calculate the hit coordinates then if the solution is in 
    # the forward direction append to the hit list.
    hit_coordinates = []
    pt1 = tuple(rpos + tmin * rdir)
    pt2 = tuple(rpos + tmax * rdir)
    
    if tmin >= 0.0:
        hit_coordinates.append(pt1)
    if tmax >= 0.0:
        hit_coordinates.append(pt2)
    return tuple(hit_coordinates)


def ray_z_cylinder(length, radius, ray_origin, ray_direction):
    """ Returns ray-cylinder intersection points for a cylinder aligned
        to the z-axis with planar end caps at z = 0.0 and z = length.
        
        Parameters
        ----------
        length : float
            The length of the cylinder
        radius : float
            The radius of the cylinder
        ray_origin : tuple of float
            The origin of th ray like, e.g. :math:`\left((0.0, 1.0, 2.0 \right)`
        ray_direction : tuple of float
            The direction **unit** vector of the ray like, e.g. :math:`(n_x, n_y, n_z)`.
        
        Returns
        -------
        intersection_info: tuple of tuple of point, distance pairs
            Returns a tuple of tuple like ((point, distance), ...) where
            point is the intersection point and distance is the distance
            from the rays origin to the intersection point. The tuple is 
            sorted by distance from the ray origin.
    
        Notes
        -----
        
        Equation of ray is [1],

        :math:`P(t) = E + t`

        where :math:`E` is the origin or "eye" point and :math:`D` is the direction vector. 
        In component form,

        .. math::

            \begin{bmatrix}
            x(t) \\
            y(t) \\
            z(t) \\ 
            \end{bmatrix} = 
            \begin{bmatrix}
            x_E + t x_D \\
            y_E + t y_D \\
            z_E + t z_D\\ 
            \end{bmatrix}

        The equation of cylinder aligned along the z direction is,

        .. math::

            x^2 + y^2 = R^2
        

        where :math`R` is the radius of the cylinder.

        Substituting the equation of the ray into the equation of the cylinder,

        .. math::
        
            (x_E + t x_D)^2 + (y_E + t y_D)^2 = R^2

        and after grouping the :math:`t^2` and :math:`t` terms,

        .. math::
        
            t^2\left(x_D^2 + y_D^2\right) + 
            t \left(2 x_E x_D + 2 y_E y _D \right) + 
            \left( x_E^2 + y_E^2 - R^2 \right) = 0

        which is a standard quadratic equation,

        .. math::
            
            at^2 + bt + c = 0

        Solution of this equation give two values :math:`\left( t_1, t_2 \right)` which 
        give the ray's distance to intersection points. To be ahead on the ray's path 
        $\left( t_1, t_2 \right) >= 0$ and to be real intersection points the values 
        must be finite and have imaginary component of zero. 

        The intersection with the cylinder caps is found by intersecting the ray with 
        two infinite planes at $z=0$ and $z=L$, where $L$ is the length of the cylinder. 
        The ray-plane intersection is given by [2],

        .. math::
        
            t = \frac{(Q - P) \cdot n}{D \cdot n}

        where :math:`t` is the distance from the ray origin to the intersection point, 
        :math:`Q` is a point on the plane and :math:`n` the **outward** facing surface 
        normal at that point. As before :math:`P` is the origin of the ray and :math:`D`
        is the ray's direction unit vector.

        For the bottom cap at :math:`z=0`,

        .. math::

            t_{\text{bot}} = 
            \frac{
            \left(
                \begin{bmatrix}
                0 \\
                0 \\
                0 \\ 
                \end{bmatrix} - 
            \begin{bmatrix}
                x_E \\
                y_E \\
                z_E \\ 
            \end{bmatrix}
            \right) \cdot 
            \begin{bmatrix}
                0 \\
                0 \\
                -1 \\ 
            \end{bmatrix}
            }{
            \begin{bmatrix}
                x_D \\
                y_D \\
                z_D \\ 
            \end{bmatrix} \cdot
            \begin{bmatrix}
                0 \\
                0 \\
                -1 \\ 
            \end{bmatrix}
            }

        and for the top cap at :math:`z=L`,

        .. math::
            t_{\text{bot}} = 
            \frac{
            \left(
                \begin{bmatrix}
                0 \\
                0 \\
                L \\ 
                \end{bmatrix} - 
            \begin{bmatrix}
                x_E \\
                y_E \\
                z_E \\ 
            \end{bmatrix}
            \right) \cdot 
            \begin{bmatrix}
                0 \\
                0 \\
                1 \\ 
            \end{bmatrix}
            }{
            \begin{bmatrix}
                x_D \\
                y_D \\
                z_D \\ 
            \end{bmatrix} \cdot
            \begin{bmatrix}
                0 \\
                0 \\
                1 \\ 
            \end{bmatrix}
            }
    

        The intersection points with :math:`t<0` and points not contained inside the circle
        of the end cap are rejected using $(x^2 + y^2) < R$, where $x$ and $y$ are the
        components of the candidate intersection point.
        
        References
        ----------
        [1] https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/
        [2] https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
        
    """
    p0 = np.array(ray_origin)
    n0 = np.array(ray_direction)
    xe, ye, ze = p0
    xd, yd, zd = n0

    # Look for intersections on the cylinder surface
    a = xd**2 + yd**2
    b = 2 * (xe*xd + ye*yd)
    c = xe**2 + ye**2 - radius**2
    tcyl = [t for t in np.roots([a, b, c]) if np.isfinite(t) and np.isreal(t) and t >= 0]
        
    # Look for intersections on the cap surfaces
    with np.errstate(divide='ignore'):
        # top cap
        point = np.array([0.0, 0.0, length])
        normal = np.array([0.0, 0.0, 1.0]) # outward facing at z = length
        ttopcap = (point - p0).dot(normal) / n0.dot(normal)
        # bottom cap
        point = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, -1.0]) # outward facing at z = 0
        tbotcap = (point - p0).dot(normal) / n0.dot(normal)
        tcap = [t for t in (tbotcap, ttopcap) if np.isfinite(t) and t >= 0.0]
    
    # Reject point cap points which are not in the cap's circle radius
    # and cylinder points which outside the length.
    cap_candidates = [(p0 + t * n0, t) for t in tcap]
    cap_candidates = [(point, t) for (point, t) in cap_candidates
                      if np.sqrt(point[0]**2 + point[1]**2) < radius]
    cyl_candidates = [(p0 + t * n0, t) for t in tcyl]
    cyl_candidates = [(point, t) for (point, t) in cyl_candidates if point[2] > 0.0 and point[2] < length]
    intersection_info = tuple(cyl_candidates) + tuple(cap_candidates)
    intersection_info = sorted(intersection_info, key=lambda pair: pair[1])
    points = tuple([tuple(p.tolist()) for p in list(zip(*intersection_info))[0]])
    return points


# Equality tests


def close_to_zero(value) -> bool:
    return np.all(np.absolute(value) < EPS_ZERO)
    

def points_equal(point1: tuple, point2: tuple) -> bool:
    return close_to_zero(distance_between(point1, point2))


def floats_close(a,b):
    return close_to_zero(a-b)


def allinrange(x, x_range):
    """ Returns True if all elements of x are inside x_range, inclusive of the 
        edge values.
        
        Parameters
        ----------
        x : array-like
            A numpy array of values.
        x_range : tuple of float
            A tuple defining a range like (xmin, xmax)
    """
    return np.where(np.logical_or(x<x_range[0], x>x_range[1]))[0].size == 0


# Vector helpers


def flip(vector):
    return -np.array(vector)


def magnitude(vector):
   return np.sqrt(np.dot(np.array(vector),np.array(vector)))


def norm(vector):
    return np.array(vector) / np.linalg.norm(vector)


def angle_between(normal, vector):
    normal = np.array(normal)
    vector = np.array(vector)
    if np.allclose(normal, vector): return 0.0
    elif np.allclose(-normal, vector): return np.pi
    dot = np.dot(normal, vector)
    return np.arccos(dot)


def smallest_angle_between(normal, vector):
    rads = angle_between(normal, vector)
    return np.arctan2(np.sin(rads), np.cos(rads))   


def distance_between(point1: tuple, point2: tuple) -> float:
    v = np.array(point1) - np.array(point2)
    d = np.linalg.norm(v)
    return d


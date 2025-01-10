from math import sqrt
import numpy as np

epsilon = 1.e-10  # Default epsilon for equality testing of points and vectors

class GeomException(Exception):
    def __init__(self, message = None):
        Exception.__init__(self, message)

def dot(v1, v2):
    """Dot product of two vectors"""
    return v1.dot(v2)

def cross(v1, v2):
    """Cross product of two vectors"""
    return v1.cross(v2)

def length(v):
    """Length of vector"""
    return sqrt(v.dot(v))

def unit(v):
    """A unit vector in the direction of v"""
    return v / length(v)

class Point(object):
    
    def __init__(self, x: float, y: float, z: float):

        if y is None and z is None:
            self.x, self.y, self.z = x
        else:
            self.x, self.y, self.z = x, y, z

    def __sub__(self, other):
        """P1 - P2 returns a vector. P - v returns a point"""
        if isinstance(other, Point):
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, Vector):
            return Point(self.x - other.dx, self.y - other.dy, self.z - other.dz)
        else:
            raise GeomException("Wrong operation.")
        
    def __add__(self, other):
        """P + v is P translated by v"""
        if isinstance(other, Vector):
            return Point(self.x + other.dx, self.y + other.dy, self.z + other.dz)
        else:
            raise GeomException("Wrong operation.")

    def __iter__(self):
        """Iterator over the coordinates"""
        return [self.x, self.y, self.z].__iter__()
    
    def __eq__(self, other):
        """Equality of points is equality of all coordinates to within 
       epsilon (defaults to 1.e-10)."""
        return (abs(self.x - other.x) < epsilon and
                 abs(self.y - other.y) < epsilon and
                 abs(self.z - other.z) < epsilon)

    def __ne__(self, other):
        """Inequality of points is inequality of any coordinates"""
        return not self.__eq__(other)
    
    def __getitem__(self, i):
        """P[i] is x, y, z for i in 0, 1, 2 resp."""
        return [self.x, self.y, self.z][i]

    def __str__(self):
        """String representation of a point"""
        return ("(%.3f,%.3f,%.3f)") % (self.x, self.y, self.z)

    def __repr__(self):
        """String representation including class"""
        return "Point" + str(self)

    def asarray(self):
        """Return as numpy array"""
        return np.array([ self.x, self.y, self.z ])


class Vector(object):
    """Represents a vector in 3-space with coordinates dx, dy, dz."""
    
    def __init__(self, dx, dy=None, dz=None):

        if dy is None and dz is None:
            self.dx, self.dy, self.dz = dx
        else:
            self.dx, self.dy, self.dz = dx, dy, dz
                              
    def __sub__(self, other):
        """Vector difference"""
        return Vector(self.dx-other.dx, self.dy-other.dy, self.dz-other.dz)

    def __add__(self, other):
        """Vector sum"""
        return Vector(self.dx+other.dx, self.dy+other.dy, self.dz+other.dz)

    def __mul__(self, scale):
        """v * r for r a float is scaling of vector v by r"""
        return Vector(scale*self.dx, scale*self.dy, scale*self.dz)

    def __rmul__(self, scale):
        """r * v for r a float is scaling of vector v by r"""
        return self.__mul__(scale)

    def __truediv__(self, scale):
        """Division of a vector by a float r is scaling by (1/r)"""
        return self.__mul__(1.0/scale)

    def __neg__(self):
        """Negation of a vector is negation of all its coordinates"""
        return Vector(-self.dx, -self.dy, -self.dz)

    def __iter__(self):
        """Iterator over coordinates dx, dy, dz in turn"""
        return [self.dx, self.dy, self.dz].__iter__()

    def __getitem__(self, i):
        """v[i] is dx, dy, dz for i in 0,1,2 resp"""
        return [self.dx, self.dy, self.dz][i]

    def __eq__(self, other):
        """Equality of vectors is equality of all coordinates to within 
       epsilon (defaults to 1.e-10)."""
        return (abs(self.dx - other.dx) < epsilon and
                 abs(self.dy - other.dy) < epsilon and
                 abs(self.dz - other.dz) < epsilon)

    def __ne__(self, other):
        """Inequality of vectors is inequality of any coordinates"""
        return not self.__eq__(other)
    
    def dot(self, other):
        """The usual dot product"""
        return self.dx*other.dx + self.dy*other.dy + self.dz*other.dz

    def cross(self, other):
        """The usual cross product"""
        return Vector(self.dy * other.dz - self.dz * other.dy,
                      self.dz * other.dx - self.dx * other.dz,
                      self.dx * other.dy - self.dy * other.dx) 

    def norm(self):
        """A normalised version of self"""
        return self/length(self)

    def asarray(self):
        """Return as numpy array"""
        return np.array([ self.dx, self.dy, self.dz ])

    def __str__(self):
        """Minimal string representation in parentheses"""
        return ("(%.7f,%.7f,%.7f)") % (self.dx, self.dy, self.dz)

    def __repr__(self):
        """String representation with class included"""
        return "Vector" + str(self)

class VectorBasis(object):
    """Represents a vector basis in 3D-space"""
    def __init__(self, v1: Vector, v2: Vector, v3: Vector):

        self.v1 = v1.norm()
        self.v2 = v2.norm()
        self.v3 = v3.norm()
        self.basis_mat = np.array( np.column_stack(( v1.asarray(), v2.asarray(), v3.asarray())) )

class Ray(object):
    """A ray is a directed line, defined by a start point and a direction"""
    
    def __init__(self, start: Point, dir: Vector):
        """Constructor takes a start point (or something convertible to point) and 
          a direction vector (which need not be normalised)."""
        self.start = start     # Ensure start point represented as a Point3
        self.dir = unit(dir)  # Direction vector

    def pos(self, t):
        """A point on a ray is start + t*dir for t positive."""
        if t >= 0:
            return self.start + t * self.dir
        else:
            raise GeomException("Attempt to obtain point not on ray")

    def __repr__(self):
        return "Ray3(%s,%s)" % (str(self.start), str(self.dir))


class Sphere(object):
    """A ray-traceable sphere"""
    
    def __init__(self, centre: Point, radius: float ):
        """Create a sphere with a given centre point and radius"""
        self.centre = centre
        self.radius = radius
        self.sqr_radius = radius * radius

    def normal(self, p):
        """The surface normal at the given point on the sphere"""
        return unit(p - self.centre)    

    def intersect(self, ray):
        """tests whether a ray intersects a sphere."""

        m = ray.start - self.centre
        b = dot( m, ray.dir )
        c = dot( m, m ) - self.sqr_radius
        if ( ( c > 0.0 ) and ( b > 0.0 ) ): return None
        discr = b*b - c
        if discr < 0.0: return None
        t = -b - np.sqrt( discr )
        return ray.pos( t )

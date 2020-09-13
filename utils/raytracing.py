import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def intersect_plane(p0: np.ndarray, n: np.ndarray, c: np.ndarray, p: np.ndarray):
    denom = np.dot(p-c,n)
    if denom > 1e-6:
        num = np.dot(p0-c,n)
        t = num/denom
        if t < 0:
            return np.inf
        return t
    else:
        return np.inf


def intersect_rectangle(rectangle: np.ndarray, ray_orgin: np.ndarray, ray_vector: np.ndarray, max_range: float):
    tri1 = ray_intersect_triangle(ray_orgin,ray_vector,rectangle[:,0:3],max_range)
    if type(tri1) == None:
        tri2 = ray_intersect_triangle(ray_orgin,ray_vector,rectangle[:,1:4],max_range)
        return tri2
    else:
        return tri1


def ray_intersect_triangle(ray_orgin: np.ndarray,ray_vector: np.ndarray, triangle:np.ndarray, max_range: float):

    ray_orgin = ray_orgin.ravel()
    ray_vector = ray_vector.ravel()
    eps = 1e-6
    vertex0 = triangle[:,0]
    vertex1 = triangle[:,1]
    vertex2 = triangle[:,2]

    edge1 = vertex1-vertex0
    edge2 = vertex2-vertex0
    h = np.cross(ray_vector,edge2)
    a = edge1.dot(h)

    if (a > -eps and a < eps):
        return np.inf*np.ones((3,1))

    f = 1.0/a
    s = ray_orgin - vertex0

    u = f*s.dot(h)

    if (u < 0.0 or u > 1.0):
        return np.inf*np.ones((3,1))

    q = np.cross(s,edge1)
    v = f*ray_vector.dot(q)
    if (v < 0.0 or u + v > 1.0):
        return np.inf*np.ones((3,1))

    t = f * edge2.dot(q)
    if t > eps and t < max_range:
        return (ray_vector*t).reshape(3,1)
    else:
        return np.inf*np.ones((3,1))


"""
vertex1  = np.array([0,0,0])
vertex2 = np.array([0,0,5])
vertex3 = np.array([5,5,0])
vertex4 = np.array([5,5,5])

verticies = np.array((vertex1,vertex2,vertex3,vertex4)).transpose()

edge1 = vertex2 - vertex1
edge2 = vertex3 - vertex1


normal = np.cross(edge1,edge2)
normal = normal/np.linalg.norm(normal)

point  = np.array([5, 5, 0])

point2 = np.array([10, 10, 10])

point3 = np.array([7,7,7])
# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)

# create x,y
xx, yy = np.meshgrid(np.linspace(0,5,10), np.linspace(0,5,10))

# calculate corresponding z
#z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

# plot the surface
fig = plt.figure()
ax = Axes3D(fig)

x = [0,5,5,0]
y = [5,0,0,5]
z = [0,0,5,5]
verts = [list(zip(x,y,z))]

ver = np.array([[0,5,5,0],[5,0,0,5],[0,0,5,5]])

ray_vector = point3-point2
ray_vector = ray_vector/np.linalg.norm(ray_vector)
p1 = intersect_rectangle(ver,point2,ray_vector)
# Add an axes
ax.add_collection3d(Poly3DCollection(verts))
# plot the surface
#ax.plot_surface(xx,yy,z, alpha=0.5)

# and plot the point
ax.scatter(point2[0] , point2[1] , point2[2],  color='green')
ax.scatter(point3[0] , point3[1] , point3[2],  color='green')
ax.scatter(p1[0],p1[1],p1[2],color='red')

ax.plot([point2[0],p1[0]],[point2[1],p1[1]],[point2[2],p1[2]])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
"""
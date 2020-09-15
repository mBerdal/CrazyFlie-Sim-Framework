import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import multiprocessing as mp
import time

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

def multi_intersect_rectangle(rectangle, ray_orgins, ray_vectors, max_range):
    tri1 = multi_ray_intersect_triangle(ray_orgins, ray_vectors, rectangle[:, 0:3], max_range)
    tm = np.any(tri1==np.inf,axis=1)
    if np.any(tm):
        tri2 = multi_ray_intersect_triangle(ray_orgins, ray_vectors, rectangle[:, 1:4], max_range)
    else:
        return tri1
    tri1[tm] = tri2[tm]
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
    h = cross_product(ray_vector,edge2)
    a = edge1.dot(h)

    if (a > -eps and a < eps):
        return np.inf*np.ones((3,1))

    f = 1.0/a
    s = ray_orgin - vertex0

    u = f*s.dot(h)

    if (u < 0.0 or u > 1.0):
        return np.inf*np.ones((3,1))

    q = cross_product(s,edge1)
    v = f*ray_vector.dot(q)
    if (v < 0.0 or u + v > 1.0):
        return np.inf*np.ones((3,1))

    t = f * edge2.dot(q)
    if t > eps and t < max_range:
        return (ray_vector*t).reshape(3,1)
    else:
        return np.inf*np.ones((3,1))


def multi_ray_intersect_triangle(ray_orgins, ray_vectors,triangle, max_range,return_t = True):
    n = ray_orgins.shape[0]
    truth_array = np.zeros(n,bool)
    eps = 1e-6
    vertex0 = triangle[:, 0]
    vertex1 = triangle[:, 1]
    vertex2 = triangle[:, 2]

    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = np.cross(ray_vectors, edge2)
    a = np.dot(h,edge1)

    tm1 = a>-eps
    tm2 = a<eps
    tm = tm1 & tm2
    truth_array[tm] = True
    a[truth_array] = np.inf
    f = 1.0/a
    s = ray_orgins - vertex0

    #u = f*np.dot(s,h)
    u = f*np.einsum('ij,ij->i', s, h)

    tm1 = u<0.0
    tm2 = u>1.0
    tm = tm1 | tm2
    truth_array[tm] = True

    q = np.cross(s,edge1)
    v = f*np.einsum('ij,ij->i', ray_vectors, q)

    tm1 = v < 0.0
    tm2 = v + u > 1.0
    tm = tm1 | tm2
    truth_array[tm] = True

    t = f * np.dot(q,edge2)

    tm1 = t < eps
    tm2 = t > max_range
    tm = tm1 | tm2
    truth_array[tm] = True

    if return_t:
        t[truth_array] = np.inf
        return t
    else:
        results = t.reshape(n,1)*ray_vectors
        results[truth_array] = np.ones(3)*np.inf
        return results




def cross_product(x,y):
    z = np.array([0,0,0],np.float)
    z[0] = x[1]*y[2] - x[2]*y[1]
    z[1] = x[2]*y[0] - x[0]*y[2]
    z[2] = x[0]*y[1] - x[1]*y[0]
    return z


def test_multiray():
    num_test = 1000000
    x1 = 5
    x2 = 5
    y1 = 0
    y2 = 5
    z1 = 0
    z2 = 3
    points = np.array([[x1, x1, x2], [y1, y1, y2], [z1, z2, z1]],np.float)

    orgins = np.random.rand(num_test,3)
    vectors = np.zeros((num_test, 3))
    for i in range(num_test):
        vec = np.random.rand(1,3)
        vectors[i,:] = vec/np.linalg.norm(vec)

    res1 = []
    start_time = time.time()
    for i in range(len(orgins)):
        res1.append(ray_intersect_triangle(orgins[i,:],vectors[i,:],points,10))
    end_time = time.time()
    print(end_time-start_time)
    res1 = np.array(res1)
    start_time = time.time()
    res2 = multi_ray_intersect_triangle(np.array(orgins,np.float),np.array(vectors,np.float),points,10)
    end_time = time.time()
    print(end_time-start_time)
    print(res1[0:10])
    print(res2[0:10])

def test_mp():
    objects = []
    points_list = []
    num_points = 10000
    for _ in range(num_points):
        x1 = np.random.randint(0, 100)
        x2 = np.random.randint(0, 100)
        y1 = np.random.randint(0, 100)
        y2 = np.random.randint(0, 100)
        z1 = np.random.randint(0, 100)
        z2 = np.random.randint(0, 100)
        points = np.array([[x1, x1, x2], [y1, y1, y2], [z1, z2, z1]])
        points_list.append(points)
        obj_tmp = {"shape": "rectangle", "points": points}
        objects.append(obj_tmp)

    ray_orgin = np.array([0,0,0])
    ray_vector = np.array([1,0,0])
    start_t = time.time()
    inter = []
    for obj in objects:
        p = ray_intersect_triangle(ray_orgin,ray_vector,obj["points"],4)
        inter.append(p)
    end_time = time.time()
    print(end_time-start_t)
    maneger = mp.Manager()

    def wrapper_intersect(points):
        return intersect_rectangle(points,ray_orgin,ray_vector,4)

    with mp.Pool(mp.cpu_count()) as pool:
        start_t = time.time()
        res = [pool.apply_async(intersect_rectangle,args=[p,ray_orgin,ray_vector,4]) for p in points]
        results = [r.get() for r in res]
    end_time = time.time()
    print(end_time - start_t)



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
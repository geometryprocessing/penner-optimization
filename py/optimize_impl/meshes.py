import numpy as np

def generate_tet():
    """
    Generate tetrahedron mesh with vertices at the origin and the three elementary basis vectors
    along with uniform target angles.

    return np.array: vertex positions
    return np.array: face indices
    return np.array: target angle values
    """
    v = np.array([[0,0,0],
                [1,0,0],
                [0,1,0],
                [0,0,1]])
    f = np.array([[0,2,1],
                [0,1,3],
                [1,2,3],
                [0,3,2]], dtype=int)
    Th_hat = np.array([np.pi]*4)
    
    return v, f, Th_hat

def generate_tri():
    """
    Generate single triangle mesh with uniform target angles.

    return np.array: vertex positions
    return np.array: face indices
    return np.array: target angle values
    """
    v = np.array([[1,0,0],
                 [0,1,0],
                 [0,0,1]])
    f = np.array([[0,1,2],], dtype=int)
    Th_hat = np.array([np.pi/3]*3)
    
    return v, f, Th_hat

def generate_nondelaunay_quad():
    """
    Generate quadrilateral mesh with nondelaunay diagonal and uniform target angles.

    return np.array: vertex positions
    return np.array: face indices
    return np.array: target angle values
    """
    v = np.array([[np.sqrt(3),0,0],
                  [0,1,0],
                  [-np.sqrt(3),0,0],
                  [0,-1,0]])
    f = np.array([[0,1,2],
                  [0,2,3]], dtype=int)
    Th_hat = np.array([np.pi/2]*4)
    
    return v, f, Th_hat

def generate_nonconvex_quad():
    """
    Generate quadrilateral mesh that is not convex.

    return np.array: vertex positions
    return np.array: face indices
    return np.array: target angle values
    """
    v = np.array([[1,0,0],
                  [2,1,0],
                  [-1,0,0],
                  [2,-1,0]])
    f = np.array([[0,1,2],
                  [0,2,3]], dtype=int)
    Th_hat = np.array([np.pi/2]*4)
    
    return v, f, Th_hat


def generate_delaunay_quad():
    """
    Generate quadrilateral mesh with delaunay diagonal and uniform target angles.

    return np.array: vertex positions
    return np.array: face indices
    return np.array: target angle values
    """
    v = np.array([[np.sqrt(3),0,0],
                  [0,1,0],
                  [-np.sqrt(3),0,0],
                  [0,-1,0]])
    f = np.array([[0,1,3],
                  [1,2,3]], dtype=int)
    Th_hat = np.array([np.pi/2]*4)

    return v, f, Th_hat

def generate_del_rect(n):
    """
    Generate Delaunay rectangle mesh
    TODO: Implement target angles

    param[in] int n: number of grid points along the x axis
    return np.array: vertex positions
    return np.array: face indices
    """
    dx = 1/(n-1)/2
    dy = 1/(n-1)

    # Generate vertices
    grid_line = np.zeros((n,3))
    grid_line[:,0] = np.linspace(0,1,n)
    strip_lines = np.concatenate((grid_line,
                                  np.array([[0,dy,0],]),
                                  grid_line+np.array([dx,dy,0])))
    strip_lines[-1,0] = 1
    v = np.concatenate((strip_lines, grid_line+np.array([0,2*dy,0])))

    # Generate faces
    f_0 = np.stack((np.arange(n-1),np.arange(1,n),np.arange(n+1,2*n))).T
    f_1 = np.stack((np.arange(n,2*n),np.arange(0,n),np.arange(n+1,2*n+1))).T
    f_2 = np.stack((np.arange(n,2*n),np.arange(n+1,2*n+1),np.arange(2*n+1,3*n+1))).T
    f_3 = np.stack((np.arange(2*n+1,3*n),np.arange(n+1,2*n),np.arange(2*n+2,3*n+1))).T
    f=np.concatenate((f_0,f_1,f_2,f_3))

    return v,f

def generate_symmetric_quad(x, y, flipped=False):
    """
    Generate symmetric quadrilateral with shear 0 given by (1,0), (-1,0), and the points
    at (x, pm y). The diagonal is along the x axis iff flipped is True

    param[in] float x: x coordinate of two quad vertices
    param[in] float y: y coordinate of two quad vertices (up to sign)
    param[in] bool flipped: use diagonal along x axis iff True
    return np.array: vertex positions
    return np.array: face indices
    return np.array: target angle values
    """
    v = np.array([[-1,     0, 0],
                  [ 1,     0, 0],
                  [ x, -y, 0],
                  [ x,  y, 0]],dtype=float)

    # Get faces for choice of diagonal
    if flipped:
        f = np.array([[0, 2, 3],
                      [3, 2, 1]],dtype=int)
    else:
        f = np.array([[0, 1, 3],
                      [0, 2, 1]],dtype=int)
    
    Th_hat = np.array([np.pi/2]*4)

    return v, f, Th_hat

def generate_two_triangle_chart(x, y, flipped=False):
    """
    Generate two triangle chart with rotational symmetry. When (x,y) is on the unit circle,
    this is the two triangle chart for some shear described in Discrete Conformal Equivalence

    param[in] float x: x coordinate of top quad vertex
    param[in] float y: y coordinate of top quad vertex
    param[in] bool flipped: use diagonal along x axis iff True
    return np.array: vertex positions
    return np.array: face indices
    return np.array: target angle values
    """
    v = np.array([[-1,     0, 0],
                  [ 1,     0, 0],
                  [-x, -y, 0],
                  [ x,  y, 0]],dtype=float)

    # Get faces for choice of diagonal
    if flipped:
        f = np.array([[0, 2, 3],
                      [3, 2, 1]],dtype=int)
    else:
        f = np.array([[0, 1, 3],
                      [0, 2, 1]],dtype=int)
    
    Th_hat = np.array([np.pi/2]*4)

    return v, f, Th_hat

def generate_two_triangle_chart_from_shear(c, flipped=False):
    """
    Generate two triangle chart with rotational symmetry described in Discrete Conformal Equivalence
    for given shear c.

    param[in] float c: shear
    param[in] bool flipped: use diagonal along x axis iff True
    return np.array: vertex positions
    return np.array: face indices
    return np.array: target angle values
    """
    r = (1 - c) / (1 + c)
    pr = np.sqrt(1 - r*r)
    return generate_two_triangle_chart(r, pr, flipped)

def generate_right_triangle(height, width):
    """
    Generate single triangle mesh with height and width specified and uniform target
    angles for a right triangle.

    param[in] height, width: triangle edge lengths
    return np.array: vertex positions
    return np.array: face indices
    return np.array: target angle values
    """
    v = np.array([[0,0,0],
                 [width,0,0],
                 [0,height,0]])
    f = np.array([[0,1,2],], dtype=int)
    Th_hat = np.array([np.pi/2, np.pi/4, np.pi/4])
    
    return v, f, Th_hat

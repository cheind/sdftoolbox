# SDF Documentation

The aim of this document is to provide a documentation of the methods implemented by this library. We limit the discussion to $\mathbb{R}^3$ Euclidean space.

## Signed distance fields

Volumentric data is commonly found in many scientific, engineering, and medical applications. Such volumentric data can be efficiently encoded using [signed distance fields](https://en.wikipedia.org/wiki/Signed_distance_function) (SDFs). A SDF is a vector valued scalar function, $f(x)$, that determines the _signed_ distance from any location to the boundary of the surface encoded by the SDF. By the properties of signed distance, one may classify a location by looking at its SDF value

$$
\begin{cases}
      f(x)>0 & \textrm{outside} \\
      f(x)=0 & \textrm{on the boundary} \\
      f(x)<0 & \textrm{inside}
\end{cases}
$$

Its useful to note that the SDF value of $(x,|f(x)|)$ can be thought of a sphere (in $\mathbb{R}^3$) centered at $x$ with radius $|f(x)|$ that touches the closest surface boundary. Frankly, it does not tell you the contact location, only the distance. Still, this property gives raise to efficient ray maching schemes for visualizing SDF volumes. Another useful property of SDFs: the gradient $\nabla_x f(x)$ points into the direction of fastest increase of signed distance.

For many primitive shapes in $\mathbb{R}^3$ analytic SDF representations are known. Additionally, one can derive modifiers, $g(x, f)$, that transform SDF in useful ways. Modifiers include rigid transformation, uniform scaling, boolean operations, repetition, displacements and many more. See [[2]](#2) for detailed information.

## Isosurface extraction

Isosurface extraction is the task of finding a suitable tesselation of the boundary (or any constant offset value) of a SDF. The methods considered in this library rely a regular SDF sampling volume from which the resulting mesh is generated. The two schemes that dominate the isosurface extraction field differ in the way they generate the tesselated topology. The following table lists the differences:

| Method |  Edge  | Face | Voxel  |
| :----: | :----: | :--: | :----: |
| Primal | Vertex | Edge | Faces  |
|  Dual  |  Face  | Edge | Vertex |

In columns are the type of intersection For example Edge means that an edge of the sampling volume crosses the boundary of the SDF. In rows are resulting elements of the mesh. For example, first row, second column means that in primal methods a vertex is created for each sampling edge that crosses the SDF boundary. See [[1]](#1) for details.

## Coordinate systems

For detailed discussions we need to define the set of coordinate systems involved. These coordinate systems match the implementation of the library.

The following image shows a sampling grid of `(3,3,3)` (blue) points. The sampling coordinates are with respect to the data coordinate system (x,y,z). The sampling points are indexed by a grid coordinate system (i,j,k) that has its origin at the mininium sampling point. Each voxel in the grid is indexed by the integer grid coordinates of its minimum (i,j,k) corner. Note that the next voxel in the i-direction of voxel (i,j,k) is (i+1,j,k).

<img src="frames.svg" width="45%">

We index edges by its source voxel index plus a label `e` $\in \{0,1,2\}$ that defines the positive edge direction. The plot below highlights a few edges.

<img src="edges.svg" width="45%">

Having three (forward) edges per voxel index allows us to easily enumerate all edges without duplicates and without missing any edges. Note, at the outer positive border faces we get a set of invalid edges (for example `(2,0,2,0)` is invalid, while `(2,0,2,1)` is valid).

## Dual Isosurface Extraction

This library implements a generic dual isosurface extraction method based. This blueprint allows the user to set different behavioral aspects to implement varios approaches (or hybrids thereof) proposed in literature.

Given a SDF and grid defining the sampling locations, the basic dual isosurface algorithm works as follows

1.  Active edges: For each edge in the sampling grid, determine if it intersects the boundary of the SDF. We call those edges with intersections _active_ edges.
1.  Edge intersection: For each active edge find the intersection point with the boundary of the surface along the edge.
1.  Vertex placement: For each grid (active) voxel with at least one active edge, determine a single vertex location.
1.  Face generation: For each active edge create a quadliteral connecting the vertices of the four active voxels sharing this active edge.

See [[3]](#3),[[1]](#1) for more information.

The library implements this recipe in vectorized form. That is, all steps of the algorithms are capable to work with multiple elements at once. This allows for a better resource usage and generally speeds up algorithmic runtime dramatically. It is also the reason that you will hardly find for-loops sprinkled all over the code.

The recipe above gives raise to different behavioral aspects that are implemented as exchangable modules:

-   _edge strategies_: determines how the intersection between an edge and the surface boundary as dictated by the SDF is found.
-   _vertex strategies_: determines how the vertex from the voxel's active edges is computed.

### Edge strategies

Edge strategies implement different methods to determine the edge/surface crossing. The following strategies are implemented

#### Linear (single-step)

This method determines the intersection by finding the root of a linear equation guided by the SDF values at the two edge endpoints. This is most commonly found method in literature. It makes the following two assumptions

1. Surface Smoothness: the SDF is assumed to be smooth
1. Small edges: the edge lengths are supposed to be small compared to size of the shape of the SDF

Together, this two assumptions lead to linearity of the surface close to edges. Hence, modelling the surface boundary using a linear equation leads to accurate estimations. Also, in case you do not have access to analytic SDFs (e.g discretized volume of SDF values) it is the best you can do.

#### Newton (iterative)

If the assumptions of the linear strategy are wrong, this leads to misplaced intersections that affect the quality of the final mesh. If you happen to have access to an analytic SDF, you might do better: We drop the assumption of surface linearity and instead find the root of the SDF along the edge iteratively. One algorithm with quadric convergence is Newton's method (it requires access to the gradient of the SDF). For our usecase (vector valued scalar function) we need a variant of it, the so called directional Newton method.

#### Bisection (iterative)

The bisection method is useful when a) linearity is not given, b) you have access to an analytic SDF and c) the gradient does not convey information along the edge direction (e.g. for some points in the SDF of a box).

## References

-   <a id="1">[1]</a>
    mikolalysenko. https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/
-   <a id="2">[2]</a>
    Inigio Quilez's.
    https://iquilezles.org/articles/distfunctions/
-   <a id="2">[3]</a> Gibson, Sarah FF. "Constrained elastic surface nets: Generating smooth surfaces from binary segmented data." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Berlin, Heidelberg, 1998.

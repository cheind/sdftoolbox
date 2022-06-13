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

Iso-surface extraction is the task of finding a suitable tesselation of the boundary (or any constant offset value) of a SDF. The methods considered in this library rely a regular SDF sampling volume from which the resulting mesh is generated. The two schemes that dominate the iso-surface extraction field differ in the way they generate the tesselated topology. The following table lists the differences:

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

## References

-   <a id="1">[1]</a>
    mikolalysenko. https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/
-   <a id="2">[2]</a>
    Inigio Quilez's.
    https://iquilezles.org/articles/distfunctions/

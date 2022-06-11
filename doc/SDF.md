# SDF Documentation

The aim of this document is to provide a documentation of the methods implemented by this library.

## Signed distance fields

Volumentric data is commonly found in many scientific, engineering, and medical applications. Such volumentric data can be efficiently encoded using [signed distance fields](https://en.wikipedia.org/wiki/Signed_distance_function) (SDFs). A SDF is a vector valued scalar function, $f(x)$, that determines the _signed_ distance from any location to the boundary of the surface encoded by the SDF. By the properties of signed distance, one may classify a location by looking at its SDF value

$$
\begin{cases}
      f(x)>0 & \textrm{outside} \\
      f(x)=0 & \textrm{on the boundary} \\
      f(x)<0 & \textrm{inside}
\end{cases}
$$

Its useful to note that the SDF value of $(x,|f(x)|)$ can be thought of a sphere (in $\mathbb{R}^3$) centered at $x$ with radius $|f(x)|$ that touches the closest surface boundary. Frankly, it does not tell you the contact location, only the distance. Still, this property gives raise to efficient ray maching schemes for visualizing SDF volumes. The gradient $\nabla_x f(x)$ points into the direction of fastest increase of signed distance.

For many primitive shapes in $\mathbb{R}^3$ analytic SDF representations are known. Additionally, one can derive modifiers, $g(x, f)$, that transform SDF in useful ways. Modifiers include rigid transformation, uniform scaling, boolean operations, repetition, displacements and many more. See [Inigio Quilez's](https://iquilezles.org/articles/distfunctions/) page for details.

## Isosurface extraction

Iso-surface extraction is the task of finding tesselation of the boundary (or any constant offset value) of a SDF.

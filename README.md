# sdftoolbox

This repository provides vectorized Python methods for creating, manipulating and tessellating signed distance fields (SDFs). This library was started to investigate variants of dual isosurface extraction methods, but has since evolved into a useful toolbox around SDFs.

<div align="center">
<img src="doc/surfacenets.svg">
</div>

The image above shows two reconstructions of a sphere displaced by waves. The reconstruction on the left uses (dual) SurfaceNets from this library, the right side shows the result of applying (primal) Marching Cubes algorithm from scikit-image.

See [examples/compare.py](examples/compare.py) for details and [doc/SDF.md](doc/SDF.md) for an in-depth documentation.

## Features

-   A generic blueprint algorithm for dual iso-surface generation from SDFs
    -   providing the following vertex placement strategies
        -   (Naive) SurfaceNets
        -   Dual Contouring
        -   Midpoint to generate Minecraft like reconstructions
    -   providing the following edge/surface boundary intersection strategies
        -   Linear (single step)
        -   Newton (iterative)
        -   Bisection (iterative)
-   Mesh postprocessing
    -   Vertex reprojection onto SDFs
    -   Quad/Triangle topology support
    -   Vertex/Face normal support
-   Tools for programmatically creating and modifying SDFs
-   Plotting support for reconstructed meshes using matplotlib
-   Exporting (STL) of tesselated isosurfaces

## Documentation

Algorithmic ideas, mathematical details and results are discussed in [doc/SDF.md](doc/SDF.md).

## Example Code

```python
# Main import
import sdftoolbox

# Setup a snowman-scene
snowman = sdftoolbox.sdfs.Union(
    [
        sdftoolbox.sdfs.Sphere.create(center=(0, 0, 0), radius=0.4),
        sdftoolbox.sdfs.Sphere.create(center=(0, 0, 0.45), radius=0.3),
        sdftoolbox.sdfs.Sphere.create(center=(0, 0, 0.8), radius=0.2),
    ],
)
family = sdftoolbox.sdfs.Union(
    [
        snowman.transform(trans=(-0.75, 0.0, 0.0)),
        snowman.transform(trans=(0.0, -0.3, 0.0), scale=0.8),
        snowman.transform(trans=(0.75, 0.0, 0.0), scale=0.6),
    ]
)
scene = sdftoolbox.sdfs.Difference(
    [
        family,
        sdftoolbox.sdfs.Plane().transform(trans=(0, 0, -0.2)),
    ]
)

# Generate the sampling locations. Here we use the default params
grid = sdftoolbox.Grid(
    res=(65, 65, 65),
    min_corner=(-1.5, -1.5, -1.5),
    max_corner=(1.5, 1.5, 1.5),
)

# Extract the surface using dual contouring
verts, faces = sdftoolbox.dual_isosurface(
    scene,
    grid,
    vertex_strategy=sdftoolbox.NaiveSurfaceNetVertexStrategy(),
    triangulate=False,
)
```

generates

<div align="center">
<img src="doc/hello_surfacenets.svg" width="30%">
</div>

See [examples/hello_dualiso.py](examples/hello_dualiso.py) for details.

## Install

Install with development extras to run all the examples.

```
pip install git+https://github.com/cheind/sdf-surfacenets#egg=sdf-surfacenets[dev]
```

## Examples

The examples can be found in [./examples/](./examples/). Each example can be invoked as a module

```
python -m examples.<name>
```

## Gallery

Here are some additional plots from the library

<div align="center">
<img src="doc/normals.gif" width="50%">
</div>
<div align="center">
<img src="doc/lods.svg" width="50%">
</div>
<div align="center">
<img src="doc/bool.svg" width="50%">
</div>
<div align="center">
<img src="doc/edge_strategies_sphere.svg" width="50%">
</div>

## References

See [doc/SDF.md](doc/SDF.md).

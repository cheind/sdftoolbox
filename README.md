# sdf-surfacenets

This repository contains a pure vectorized Python implementation of (naive) SurfaceNets for the problem of isosurface extraction from signed distance fields (SDF).

<div align="center">
<img src="doc/surfacenets.svg">
</div>

The above image shows two reconstructions of a sphere superimposed by waves. The reconstruction on the left uses SurfaceNets from this library, the right side shows a Marching Cubes mesh generated using scikit-image.

## Features

-   A vectorized SurfaceNets method
-   Midpoint and naive surface point placement
-   Quad/Triangle support
-   Support for creating and manipulating SDFs
-   Plotting support for reconstructed meshes
-   Export (STL) support for reconstruted meshes

## Example Code

```python
# Main import
import surfacenets as sn

# Setup the scene
scene = sn.sdfs.Union(
    [
        sn.sdfs.Sphere.create(center=(0, 0, 0), radius=0.5),
        sn.sdfs.Sphere.create(center=(0, 0, 0.6), radius=0.3),
        sn.sdfs.Sphere.create(center=(0, 0, 1.0), radius=0.2),
    ],
    alpha=8,
)

# Get some sampling locations
xyz, spacing = sn.sample_volume()

# Evaluate the SDF
sdfv = scene.sample(xyz)

# Extract the surface using quadliterals
verts, faces = sn.surface_nets(
    sdfv,
    spacing=spacing,
    vertex_placement_mode="naive",
    triangulate=False,
)
verts += xyz[0, 0, 0]
```

See [examples/hello_surfacenets.py](examples/hello_surfacenets.py) for details.

## References

-   Gibson, S. F. F. (1999). Constrained elastic surfacenets: Generating smooth models from binary segmented data. TR99, 24.
-   Naive SurfaceNets: https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/
-   Signed Distance Fields: https://iquilezles.org/articles/distfunctions/

Basic isosurface extraction from signed distance functions using naive surface nets in Python

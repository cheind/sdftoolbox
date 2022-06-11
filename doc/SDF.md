# SDF Documentation

The aim of this document is to provide a documentation of the methods implemented by this library.

## Signed distance fields

A [signed distance field](https://en.wikipedia.org/wiki/Signed_distance_function), $f(x)$, is a vector valued scalar function that computes the signed distance from any location to the boundary. One may classify any location by looking at its SDF

$$
\begin{cases}
      f(x)>0 & \textrm{outside} \\
      f(x)=0 & \textrm{on the boundary} \\
      f(x)<0 & \textrm{inside}
\end{cases}
$$

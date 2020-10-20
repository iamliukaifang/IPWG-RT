# Implementation of Weak Galerkin FEM with RT element

**Purpose**: A Python implementation of WG method with RT element for Poisson equation:
$$
-\Delta u = f, \quad \text{ in } \Omega,\\
u =g, \quad \text{ on }\partial \Omega.
$$
**Required libraries**: `numpy`, `matplotlib`,`scipy`,`numba`, optional libs: `triangles`.
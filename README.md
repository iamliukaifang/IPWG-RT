# Implementation of IPWG FEM with RT element

**Purpose**: A Python implementation of IPWG method with RT element for Poisson equation:
$$
-\Delta u = f, \quad \text{ in } \Omega,\\
u =g, \quad \text{ on }\partial \Omega.
$$
**Required libraries**: `numpy`, `matplotlib`,`scipy`,`numba`, optional libs: `triangles`.

## How to use?

Open the main file `IPWG-RT.py`:

* change PDE example by `import elliptic_equation_ex1 as pde`;
* change parameters of IPWG method in pde files, e.g., in `elliptic_equation_ex1.py`:
    ```Python
    order = 0       # degree of element, 0 means k=0 in (Pk,Pk, RTk);
    EPSILON = -1.	# -1=>SIPG, 0=>IIPG, 1=>NIPG
    SIGMA = 1.
    BETA = 1
    ```

* output are errors in H1 and L2 norms;

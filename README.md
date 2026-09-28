#acou3d

These are some tests of acoustic modeling in 3D.


* `python-pyvista` attempts to model a speaker enclosure and radiating acoustic field, with acoustic interaction between the driver, cabinet and surrounding air (linear BEM) using bempp-cl, gmsh, matplotlib, numba, numpy, pyopencl, pyvista, 
scipy, and vtk.

  ![python-pyvista BEM demo: cabinet surface pressure, directivity balloon and SPL response](docs/python-pyvista.png)

---

* `python-taichi` is a physics-based air-particle simulator, currently showing a monopole point source's transient impact on 64k volumetric particles.  Taichi also runs on GPU if available.

  ![python-taichi slice view: a 500 Hz burst radiating from the central source](docs/python-taichi.png)

  ![python-taichi 3D view: the same burst in the full 64k-particle cloud](docs/python-taichi-3d.png)



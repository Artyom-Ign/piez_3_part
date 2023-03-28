#!/usr/bin/python3
#do some changes
import ufl
import numpy as np
import dolfinx
from dolfinx import la,log
import dolfinx.plot as plot
from dolfinx.fem import (Expression, Function, FunctionSpace, Constant,
                         VectorFunctionSpace, dirichletbc, form,
                         locate_dofs_topological)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, GhostMode, create_box, create_unit_cube,
                          locate_entities_boundary)

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from ufl import ds, dx, grad, inner, Identity, tr, ln, derivative, det
# import pyvistaqt as pvqt
import pyvista as pv



# mesh = RectangleMesh(
#     MPI.COMM_WORLD,
#     [np.array([0, 0, 0]), np.array([1, 1, 0])], [32, 32],
#     CellType.triangle, dolfinx.cpp.mesh.GhostMode.none)
log.set_output_file("log.txt")
# log.
N1 = 16
mesh = create_unit_cube(MPI.COMM_WORLD, N1, N1, N1)

V = VectorFunctionSpace(mesh, ("Lagrange", 1))

# u0 = Function(V)
# with u0.vector.localForm() as u0_loc:
#     u0_loc.set(0)

u, v = Function(V), ufl.TestFunction(V)

# facets = locate_entities_boundary(mesh, 1,
#                                   lambda x: np.logical_or(np.isclose(x[2], 1),
#                                                           np.isclose(x[2], 1)))
facets = locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[2], 0))


bc = dirichletbc(u, locate_dofs_topological(V, 1, facets))


# x = ufl.SpatialCoordinate(mesh)
I = Identity(mesh.topology.dim)
F = I + grad(u)
C = F.T * F
J = det(F)
Ic = tr(C)*J**(-2/3)

#f_grav = Constant(mesh, [0.,0, -1. ])
# psi = 0.5 * (Ic - 3) - ln(J) + (500 / 2) * (ln(J)) ** 2
psi = 0.5*(Ic - 3) + 200*(J - 1)**2
Pi = psi*dx + 0.1*u[2]*dx  # inner(f_grav, u)*dx
FF = derivative(Pi, u, v)
problem = NonlinearProblem(FF, u, bcs=[bc],) # petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# uh = problem.solve()
solver = NewtonSolver(MPI.COMM_WORLD, problem)
r = solver.solve(u)
# f_grav

with XDMFFile(MPI.COMM_WORLD, "elast_x.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u)

# print(u([0.5, 0.5, 0.5]))
uh = u
# Update ghost entries and plot
# uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cell_entities = np.arange(num_cells, dtype=np.int32)

topology, cell_types, x = plot.create_vtk_mesh(V)
grid = pv.UnstructuredGrid(topology, cell_types, x)
num_dofs = x.shape[0]
values = np.zeros((num_dofs, 3), dtype=np.float64)
values[:, :mesh.geometry.dim] = u.x.array.reshape(num_dofs, V.dofmap.index_map_bs)


# Create a point cloud of glyphs
# grid = pv.UnstructuredGrid(topology, cell_types, geometry)
grid["vectors"] = values
grid.set_active_vectors("vectors")



plotter = pv.Plotter()
# plotter.add_mesh(grid, show_edges=True)
warped = grid.warp_by_vector()
# warped = grid.warp_by_scalar()
plotter.add_mesh(warped)

# If pyvista environment variable is set to off-screen (static) plotting save png
# if pyvista.OFF_SCREEN:
#     pyvista.start_xvfb(wait=0.1)
#     plotter.screenshot("uh.png")
# else:
plotter.show()



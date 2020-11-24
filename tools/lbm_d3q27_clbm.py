# Three-dimensional cascaded lattice Boltzmann method: Improved implementation and consistent forcing scheme

import sys
import math
import sympy
from sympy import Matrix, diag, eye, ones, zeros, symbols, pprint
from code_printer import CodePrinter

sympy.init_printing(use_unicode=True, num_columns=220, wrap_line=False)

src = CodePrinter('computeCLBM')

"""Kernel input constants"""
# Lattice coordinate
x = symbols('x')
y = symbols('y')
z = symbols('z')
# Lattice size
nx = symbols('nx')
ny = symbols('ny')
nz = symbols('nz')
# Kinematic viscosity
nu = symbols('nu')
# Thermal diffusivity
nuT = symbols('nuT')
# Smagorinsky constant
C = symbols('C')
# Turbulent Prandtl number
Pr_t = symbols('Pr_t')
# Gravity times thermal expansion
gBetta = symbols('gBetta')
# Reference temperature for Boussinesq
Tref = symbols('Tref')

"""Kernel input distribution functions"""
# Velocity PDFs
fi = Matrix([symbols(f'f{i}') for i in range(0, 27)])
Ti_tilde = Matrix([symbols(f'Ti_tilde{i}') for i in range(0, 27)])
df_tmp = symbols('df_tmp')
dfT_tmp = symbols('dfT_tmp')
phy = symbols('phy')
rho = symbols('rho')

src.parameter(x, y, z, nx, ny, nz, type='int')
src.parameter(nu, nuT, C, Pr_t, gBetta, Tref, fi)
src.parameter(df_tmp, type='real_t* __restrict__ ')
src.parameter(dfT_tmp, type='real_t* __restrict__ ')
src.parameter(phy, type='PhysicalQuantity*')

ux = symbols('ux')
uy = symbols('uy')
uz = symbols('uz')

src.define(Matrix([ux, uy, uz]), rho, Ti_tilde)

"""Kernel generation constants"""
# Lattice time step
dt = 1.0
# Lattice grid size
dx = dy = dz = 1.0
# LES filter width
dfw = 2.0*(dx*dy*dz)**(1.0/3.0)
# Mean density in system
rho_0 = 1.0
# Speed of sound squared
cs2 = 1.0/3.0

ei = Matrix([
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [-1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, -1, 1],
    [0, 1, -1],
    [0, -1, -1],
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [1, 1, -1],
    [-1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1],
])


def phi(ei):
    ex = ei[0]
    ey = ei[1]
    ez = ei[2]
    p0 = ex**0
    p1 = ex
    p2 = ey
    p3 = ez
    p4 = ex*ey
    p5 = ex*ez
    p6 = ey*ez
    p7 = ex*ex
    p8 = ey*ey
    p9 = ez*ez
    p10 = ex*ey*ey
    p11 = ex*ez*ez
    p12 = ey*ex*ex
    p13 = ez*ex*ex
    p14 = ey*ez*ez
    p15 = ez*ey*ey
    p16 = ex*ey*ez
    p17 = ex*ex*ey*ey
    p18 = ex*ex*ez*ez
    p19 = ey*ey*ez*ez
    p20 = ex*ex*ey*ez
    p21 = ex*ey*ey*ez
    p22 = ex*ey*ez*ez
    p23 = ex*ey*ey*ez*ez
    p24 = ex*ex*ey*ez*ez
    p25 = ex*ex*ey*ey*ez
    p26 = ex*ex*ey*ey*ez*ez
    return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26]


M = Matrix([phi(ei.row(i)) for i in range(0, 27)]).transpose()
m = M*fi

# ux = symbols('ux')
# uy = symbols('uy')
# uz = symbols('uz')

src.comment('Macroscopic density')
src.let(rho, m[0])
src.comment('Macroscopic velocity')
src.let(ux, m[1]/rho)
src.let(uy, m[2]/rho)
src.let(uz, m[3]/rho)
U = Matrix([[ux, uy, uz]])

N = Matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-ux, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-uy, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-uz, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ux * uy, -uy, -ux, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ux * uz, -uz, 0, -ux, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [uy * uz, 0, -uz, -uy, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ux * ux, -2 * ux, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [uy * uy, 0, -2 * uy, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [uz * uz, 0, 0, -2 * uz, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-ux * uy * uy, uy * uy, 2 * ux * uy, 0, -2 * uy, 0, 0, 0, -ux, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0],
            [-ux * uz * uz, uz * uz, 0, 2 * ux * uz, 0, -2 * uz, 0, 0, 0, -ux, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0],
            [-ux * ux * uy, 2 * ux * uy, ux * ux, 0, -2 * ux, 0, 0, -uy, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0],
            [-ux * ux * uz, 2 * ux * uz, 0, ux * ux, 0, -2 * ux, 0, -uz, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0],
            [-uy * uz * uz, 0, uz * uz, 2 * uy * uz, 0, 0, -2 * uz, 0, 0, -uy, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0],
            [-uy * uy * uz, 0, 2 * uy * uz, uy * uy, 0, 0, -2 * uy, 0, -uz, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0],
            [-ux * uy * uz, uy * uz, ux * uz, ux * uy, -uz, -uy, -ux, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0],
            [ux * ux * uy * uy, -2 * ux * uy * uy, -2 * uy * ux * ux, 0, 4 * ux * uy, 0, 0, uy * uy, ux * ux, 0, -2 *
             ux, 0, -2 * uy, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ux * ux * uz * uz, -2 * ux * uz * uz, 0, -2 * uz * ux * ux, 0, 4 * ux * uz, 0, uz * uz, 0, ux * ux, 0, -2 *
             ux, 0, -2 * uz, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [uy * uy * uz * uz, 0, -2 * uy * uz * uz, -2 * uz * uy * uy, 0, 0, 4 * uy * uz, 0, uz * uz, uy * uy, 0, 0,
             0, 0, -2 * uy, -2 * uz, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [ux * ux * uy * uz, -2 * ux * uy * uz, -ux * ux * uz, -ux * ux * uy, 2 * ux * uz, 2 * ux * uy, ux * ux, uy *
             uz, 0, 0, 0, 0, -uz, -uy, 0, 0, -2 * ux, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [ux * uy * uy * uz, -uy * uy * uz, -2 * ux * uy * uz, -ux * uy * uy, 2 * uy * uz, uy * uy, 2 * ux * uy, 0,
             ux * uz, 0, -uz, 0, 0, 0, 0, -ux, -2 * uy, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [ux * uy * uz * uz, -uy * uz * uz, -ux * uz * uz, -2 * ux * uy * uz, uz * uz, 2 * uy * uz, 2 * ux * uz, 0,
             0, ux * uy, 0, -uy, 0, 0, -ux, 0, -2 * uz, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [-ux * uy * uy * uz * uz, uy * uy * uz * uz, 2 * ux * uy * uz * uz, 2 * ux * uy * uy * uz, -2 * uy * uz *
             uz, -2 * uy * uy * uz, -4 * ux * uy * uz, 0, -ux * uz * uz, -ux * uy * uy, uz * uz, uy * uy, 0, 0, 2 * ux
             * uy, 2 * ux * uz, 4 * uy * uz, 0, 0, -ux, 0, -2 * uz, -2 * uy, 1, 0, 0, 0],
            [-ux * ux * uy * uz * uz, 2 * ux * uy * uz * uz, ux * ux * uz * uz, 2 * ux * ux * uy * uz, -2 * ux * uz *
             uz, -4 * ux * uy * uz, -2 * ux * ux * uz, -uy * uz * uz, 0, -ux * ux * uy, 0, 2 * ux * uy, uz * uz, 2 * uy
             * uz, ux * ux, 0, 4 * ux * uz, 0, -uy, 0, -2 * uz, 0, -2 * ux, 0, 1, 0, 0],
            [-ux * ux * uy * uy * uz, 2 * ux * uy * uy * uz, 2 * ux * ux * uy * uz, ux * ux * uy * uy, -4 * ux * uy *
             uz, -2 * ux * uy * uy, -2 * ux * ux * uy, -uy * uy * uz, -ux * ux * uz, 0, 2 * ux * uz, 0, 2 * uy * uz, uy
             * uy, 0, ux * ux, 4 * ux * uy, -uz, 0, 0, -2 * uy, -2 * ux, 0, 0, 0, 1, 0],
            [ux * ux * uy * uy * uz * uz, -2 * ux * uy * uy * uz * uz, -2 * ux * ux * uy * uz * uz, -2 * ux * ux * uy *
             uy * uz, 4 * ux * uy * uz * uz, 4 * ux * uy * uy * uz, 4 * ux * ux * uy * uz, uy * uy * uz * uz, ux * ux *
             uz * uz, ux * ux * uy * uy, -2 * ux * uz * uz, -2 * ux * uy * uy, -2 * uy * uz * uz, -2 * uy * uy * uz, -2
             * ux * ux * uy, -2 * ux * ux * uz, -8 * ux * uy * uz, uz * uz, uy * uy, ux * ux, 4 * uy * uz, 4 * ux * uz,
             4 * ux * uy, -2 * ux, -2 * uy, -2 * uz, 1]])

# Central moment set
src.let(Ti_tilde, N*M*fi)

src.generate(sys.argv)

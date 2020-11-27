"""
Kernel for setting initial conditions of LBM lattice to equlilibrium values.
"""

import sys
import math
import sympy
from sympy import Matrix, diag, eye, ones, zeros, symbols, pprint
from code_printer import CodePrinter

sympy.init_printing(use_unicode=True, num_columns=220, wrap_line=False)

src = CodePrinter('InitKernel', prefix='__global__', is_header=False)

"""Kernel input constants"""
# Lattice size
nx = symbols('nx')
ny = symbols('ny')
nz = symbols('nz')
# Initial density
rho = symbols('rho')
# Initial velocity
vx = symbols('vx')
vy = symbols('vy')
vz = symbols('vz')
# Initial temperature
T = symbols('T')
# Velocity square term
sq_term = symbols('sq_term')

"""Kernel input distribution functions"""
df = symbols('df')
dfT = symbols('dfT')

src.parameter(df, type='real_t* __restrict__ ')
src.parameter(dfT, type='real_t* __restrict__ ')
src.parameter(nx, ny, nz, type='int')
src.parameter(rho, vx, vy, vz, T, type='real_t')
V = Matrix([vx, vy, vz])
# Speed of sound squared
cs2 = 1.0/3.0

src.define(sq_term)
src.append('Vector3<int> pos(threadIdx.x, blockIdx.x, blockIdx.y);')
src.append('if ((pos.x() >= nx) || (pos.y() >= ny) || (pos.z() >= nz)) return;')
src.append('const int x = pos.x();')
src.append('const int y = pos.y();')
src.append('const int z = pos.z();')


# LBM velocity vectors for D3Q19 (and D3Q7)
ei = Matrix([
    [0, 0, 0],    # 0
    [1, 0, 0],    # 1
    [-1, 0, 0],   # 2
    [0, 1, 0],    # 3
    [0, -1, 0],   # 4
    [0, 0, 1],    # 5
    [0, 0, -1],   # 6
    [1, 1, 0],    # 7
    [-1, -1, 0],  # 8
    [1, -1, 0],   # 9
    [-1, 1, 0],   # 10
    [1, 0, 1],    # 11
    [-1, 0, -1],  # 12
    [1, 0, -1],   # 13
    [-1, 0, 1],   # 14
    [0, 1, 1],    # 15
    [0, -1, -1],  # 16
    [0, 1, -1],   # 17
    [0, -1, 1]    # 18
])

# Density weighting factors for D3Q19 velocity PDFs
e_omega = Matrix([
    1.0/3.0,
    1.0/18.0,
    1.0/18.0,
    1.0/18.0,
    1.0/18.0,
    1.0/18.0,
    1.0/18.0,
    1.0/36.0,
    1.0/36.0,
    1.0/36.0,
    1.0/36.0,
    1.0/36.0,
    1.0/36.0,
    1.0/36.0,
    1.0/36.0,
    1.0/36.0,
    1.0/36.0,
    1.0/36.0,
    1.0/36.0
])

# Density weighting factors for D3Q7 energy PDFs
e_omegaT = Matrix([
    0.0,
    1.0/6.0,
    1.0/6.0,
    1.0/6.0,
    1.0/6.0,
    1.0/6.0,
    1.0/6.0
])

src.let(sq_term,  -1.5*(vx*vx + vy*vy + vz*vz))


def phi_eq(ei, omega):
    return omega*rho*(1.0 + ei.dot(V)/cs2 + (ei.dot(V)**2)/(2.0*cs2**2) + sq_term)


def chi_eq(ei):
    b = 7.0
    return T/b*(1.0 + b/2.0*ei.dot(V))


df3D = Matrix([phi_eq(ei.row(i), e_omega.row(i))[0] for i in range(0, 19)])
Tdf3D = Matrix([chi_eq(ei.row(i)) for i in range(0, 7)])

for i in range(0, 19):
    src.append(f'df3D({i}, x, y, z, nx, ny, nz) = {src.eval(df3D.row(i)[0])};')

for i in range(0, 7):
    src.append(
        f'Tdf3D({i}, x, y, z, nx, ny, nz) = {src.eval(Tdf3D.row(i)[0])};')

src.include("InitKernel.hpp")

src.generate(sys.argv)

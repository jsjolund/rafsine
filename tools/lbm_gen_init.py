"""
Kernel for setting initial conditions of LBM lattice to equlilibrium values.
"""

import sys
import sympy
from sympy import Matrix, diag, eye, ones, zeros, symbols, pprint
from code_printer import CppFile

from ddqq import ei, d3q27_weights, d3q19_weights, d3q7_weights

sympy.init_printing(use_unicode=True, num_columns=220, wrap_line=False)

src = CppFile('InitKernel')
src.comment("Generated by lbm_gen_init.py")
src.include("InitKernel.hpp")

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
# dfT = symbols('dfT')

src.parameter(df, type='real_t* __restrict__ ')
# src.parameter(dfT, type='real_t* __restrict__ ')
src.parameter(nx, ny, nz, type='int')
src.parameter(rho, vx, vy, vz, T, type='real_t')
V = Matrix([vx, vy, vz])
# Speed of sound squared
cs2 = 1.0/3.0

src.define(sq_term)
src.let(sq_term, -1.5*(vx*vx + vy*vy + vz*vz))
src.append('Vector3<int> pos(threadIdx.x, blockIdx.x, blockIdx.y);')
src.append('if ((pos.x() >= nx) || (pos.y() >= ny) || (pos.z() >= nz)) return;')
src.append('const int x = pos.x();')
src.append('const int y = pos.y();')
src.append('const int z = pos.z();')
src.append('switch (Q) {')


# Velocity equilibrium distribution function
def phi_eq(ei, omega):
    return omega*rho*(1.0 + ei.dot(V)/cs2 + (ei.dot(V)**2)/(2.0*cs2**2) + sq_term)


# Temperature equilibrium distribution function
def chi_eq(ei):
    b = 7.0
    return T/b*(1.0 + b/2.0*ei.dot(V))

for q in (7, 19, 27):
    weights = 0
    df3D = 0
    if q == 7:
        weights = d3q7_weights
        df3D = Matrix([chi_eq(ei.row(i)) for i in range(0, 7)])
    elif q == 19:
        weights = d3q19_weights
        df3D = Matrix([phi_eq(ei.row(i), weights.row(i))[0] for i in range(0, q)])
    elif q == 27:
        weights = d3q27_weights
        df3D = Matrix([phi_eq(ei.row(i), weights.row(i))[0] for i in range(0, q)])
    else:
        print('invalid q')
    src.append(f'case {q}:')
    for i in range(0, q):
        src.append(f'df[I4D({i}, x, y, z, nx, ny, nz)] = {src.eval(df3D.row(i)[0])};')
    src.append('break;')

src.append('}')

src.set_prefix('template <int Q>\n__global__')

for q in (7, 19, 27):
    src.appendix(f'template __global__ void InitKernel<{q}>(real_t* __restrict__ df, int nx, int ny, int nz, float rho, float vx, float vy, float vz, float T);\n')

src.generate(sys.argv)

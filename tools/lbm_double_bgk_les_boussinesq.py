"""BGK-LBM kernel generator for D3Q19 velocity and D3Q7 temperature distributios.
Implements LES for turbulence modelling and Boussinesq approximation of body force.
"""

import sys
import math
import sympy
from sympy import Matrix, diag, eye, ones, zeros, symbols, pprint
from code_printer import CodePrinter

sympy.init_printing(use_unicode=True, num_columns=220, wrap_line=False)

src = CodePrinter('computeBGK')

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
fi = Matrix([symbols(f'f{i}') for i in range(0, 19)])
# Temperature PDFs
Ti = Matrix([symbols(f'T{i}') for i in range(0, 7)])
df_tmp = symbols('df_tmp')
dfT_tmp = symbols('dfT_tmp')
phy = symbols('phy')

src.parameter(x, y, z, nx, ny, nz, type='int')
src.parameter(nu, nuT, C, Pr_t, gBetta, Tref, fi, Ti)
src.parameter(df_tmp, type='real_t* __restrict__ ')
src.parameter(dfT_tmp, type='real_t* __restrict__ ')
src.parameter(phy, type='PhysicalQuantity*')

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

"""Temporary variables"""
fi_eq = Matrix([symbols(f'f{i}eq') for i in range(0, 19)])
fi_neq = Matrix([symbols(f'f{i}neq') for i in range(0, 19)])
Ti_eq = Matrix([symbols(f'T{i}eq') for i in range(0, 7)])

Sxx = symbols('Sxx')
Syy = symbols('Syy')
Szz = symbols('Szz')
Sxy = symbols('Sxy')
Syz = symbols('Syz')
Sxz = symbols('Sxz')
S_bar = symbols('S_bar')
ST = symbols('ST')

rho = symbols('rho')
sq_term = symbols('sq_term')
tau_V = symbols('tau_V')
tau_T = symbols('tau_T')

T = symbols('T')
vx = symbols('vx')
vy = symbols('vy')
vz = symbols('vz')

Fup = symbols('Fup')
Fdown = symbols('Fdown')

src.define(fi_eq, fi_neq, Ti_eq,  S_bar, ST, rho, sq_term, T,
           Matrix([Sxx, Syy, Szz, Sxy, Syz, Sxz]),
           Matrix([tau_V, tau_T]),
           Matrix([Fup, Fdown]),
           Matrix([vx, vy, vz]))

"""Kernel generator"""
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


# Transformation matrix for transition from velocity to moment space
def phi(ei):
    p0 = ei.norm()**0
    p1 = ei[0]
    p2 = ei[1]
    p3 = ei[2]
    return [p0, p1, p2, p3]


M = Matrix([phi(ei.row(i)) for i in range(0, 19)]).transpose()
m = M*fi
src.comment('Macroscopic density')
src.let(rho, m[0])
src.comment('Macroscopic velocity')
src.let(vx, m[1]/rho)
src.let(vy, m[2]/rho)
src.let(vz, m[3]/rho)
V = Matrix([[vx, vy, vz]])
src.comment('Macroscopic temperature')
src.let(T, Ti.dot(ones(1, 7)))

src.let(sq_term, -1.0*V.dot(V)/(2.0*cs2))


def phi_eq(ei, omega):
    return omega*rho*(1.0 + ei.dot(V)/cs2 + (ei.dot(V)**2)/(2.0*cs2**2) + sq_term)


src.comment('Compute the equilibrium distribution function')
f_eq = Matrix([phi_eq(ei.row(i), e_omega.row(i)) for i in range(0, 19)])
src.let(fi_eq, f_eq)


def chi_eq(ei):
    b = 7.0
    return T/b*(1.0 + b/2.0*ei.dot(V))


src.comment('Temperature equilibirum distribution functions')
T_eq = Matrix([chi_eq(ei.row(i)) for i in range(0, 7)])
src.let(Ti_eq, T_eq)


# Boussinesq approximation of body force
def Fi(i):
    return (T - Tref)*gBetta*ei.row(i)


src.comment('Boussinesq approximation of body force')
src.let(Fup, Fi(5)[2])
src.let(Fdown, Fi(6)[2])
Fi = zeros(19, 1)
Fi[5] = Fup
Fi[6] = Fdown

src.comment('Difference to velocity equilibrium')
src.let(fi_neq, fi - fi_eq)


def Sij(alpha, beta):
    return Matrix([ei.row(i)[alpha]*ei.row(i)[beta]*fi_neq[i] for i in range(0, 19)])


src.comment('Non equilibrium stress-tensor for velocity')
src.let(Sxx, Sij(0, 0).dot(ones(1, 19)))
src.let(Syy, Sij(1, 1).dot(ones(1, 19)))
src.let(Szz, Sij(2, 2).dot(ones(1, 19)))
src.let(Sxy, Sij(0, 1).dot(ones(1, 19)))
src.let(Sxz, Sij(0, 2).dot(ones(1, 19)))
src.let(Syz, Sij(1, 2).dot(ones(1, 19)))

src.comment('Magnitude of strain rate tensor')
src.let(S_bar, ((Sxx*Sxx + Syy*Syy + Szz*Szz +
                 2.0*(Sxy*Sxy + Syz*Syz + Sxz*Sxz)))**(1.0/2.0))
src.let(ST, (1.0 / 6.0) * ((nu**2 + 18.0 * C**2 * S_bar)**(1.0/2.0) - nu))

src.comment('Modified relaxation time')
src.let(tau_V, 3.0*(nu + ST) + 0.5)

src.comment('Modified relaxation time for the temperature')
src.let(tau_T, 3.0*(nuT + ST/Pr_t) + 0.5)

omega = Matrix([-1.0/tau_V*fi + 1.0/tau_V*fi_eq])

omegaT = Matrix([-1.0/tau_T*Ti + 1.0/tau_T*Ti_eq])

# Write distribution functions
dftmp3D = Matrix([[fi.row(i)[0] + omega.row(i)[0] + Fi.row(i)[0]]
                  for i in range(0, 19)])

Tdftmp3D = Matrix([[Ti.row(i)[0] + omegaT.row(i)[0]] for i in range(0, 7)])

src.comment('Relax velocity')
for i in range(0, 19):
    src.append(f'dftmp3D({i}, x, y, z, nx, ny, nz) = {dftmp3D.row(i)[0]};')

src.comment('Relax temperature')
for i in range(0, 7):
    src.append(f'Tdftmp3D({i}, x, y, z, nx, ny, nz) = {Tdftmp3D.row(i)[0]};')

src.comment('Store macroscopic values')
src.append('phy->rho = rho;')
src.append('phy->T = T;')
src.append('phy->vx = vx;')
src.append('phy->vy = vy;')
src.append('phy->vz = vz;')

src.include("CudaUtils.hpp")
src.include("PhysicalQuantity.hpp")

src.handle(sys.argv)

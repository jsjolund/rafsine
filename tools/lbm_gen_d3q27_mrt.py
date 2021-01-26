"""
A D3Q27 multiple-relaxation-time lattice Boltzmann method for turbulent flows
K. Suga, Y. Kuwata, K. Takashima, R. Chikasue
"""

import sys
import sympy
from sympy import Matrix, ones, zeros, symbols, GramSchmidt
from code_printer import HppFile

from ddqq import ei, d3q27_weights

sympy.init_printing(use_unicode=True, num_columns=220, wrap_line=False)

src = HppFile('computeMRT27')
src.header(f'// Generated by {sys.argv[0]}')
src.include('CudaUtils.hpp')
src.include('PhysicalQuantity.hpp')

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
mi_eq = Matrix([symbols(f'm{i}eq') for i in range(0, 27)])
mi_eq[0] = 0
mi_diff = Matrix([symbols(f'm{i}diff') for i in range(0, 27)])
mi_diff[0] = 0
omega = Matrix([symbols(f'omega{i}') for i in range(0, 27)])

Sxx = symbols('Sxx')
Syy = symbols('Syy')
Szz = symbols('Szz')
Sxy = symbols('Sxy')
Syz = symbols('Syz')
Sxz = symbols('Sxz')
S_bar = symbols('S_bar')
ST = symbols('ST')

rho = symbols('rho')
jx = symbols('jx')
jy = symbols('jy')
jz = symbols('jz')
tau_V = symbols('tau_V')
tau_T = symbols('tau_T')

T = symbols('T')
ux = symbols('ux')
uy = symbols('uy')
uz = symbols('uz')
u_bar = symbols('u_bar')

Fup = symbols('Fup')
Fdown = symbols('Fdown')

Ti_eq = Matrix([symbols(f'T{i}eq') for i in range(0, 7)])

src.define(mi_eq, mi_diff, omega, S_bar, ST,
           T, Fup, Fdown, tau_V, tau_T,
           Matrix([Sxx, Syy, Szz, Sxy, Syz, Sxz]),
           Matrix([rho, jx,  jy, jz]),
           Matrix([ux, uy, uz]), u_bar, Ti_eq)


###############################################################################
# Velocity distribution functions

def phi(ei):
    # Transformation matrix for transition from velocity to moment space
    ex = ei[0]
    ey = ei[1]
    ez = ei[2]
    # Density
    p0 = 1
    # Momentum
    p1 = ex
    p2 = ey
    p3 = ez
    # Kinetic energy
    p4 = ex**2 + ey**2 + ez**2
    # Second order tensor
    p5 = 2*ex**2 - ey**2 - ez**2
    p6 = ey**2 - ez**2
    p7 = ex*ey
    p8 = ey*ez
    p9 = ez*ex
    # Fluxes of the energy and square of energy
    p10 = 3*(ex**2 + ey**2 + ez**2)*ex
    p11 = 3*(ex**2 + ey**2 + ez**2)*ey
    p12 = 3*(ex**2 + ey**2 + ez**2)*ez
    p13 = 9/2*(ex**2 + ey**2 + ez**2)**2*ex
    p14 = 9/2*(ex**2 + ey**2 + ez**2)**2*ey
    p15 = 9/2*(ex**2 + ey**2 + ez**2)**2*ez
    # Square and cube of energy
    p16 = 3/2*(ex**2 + ey**2 + ez**2)**2
    p17 = 9/2*(ex**2 + ey**2 + ez**2)**3
    # Product of the second order tensor and the energy,
    p18 = 3*(2*ex**2 - ey**2 - ez**2)*(ex**2 + ey**2 + ez**2)
    p19 = 3*(ey**2 - ez**2)*(ex**2 + ey**2 + ez**2)
    p20 = 3*ex*ey*(ex**2 + ey**2 + ez**2)
    p21 = 3*ey*ez*(ex**2 + ey**2 + ez**2)
    p22 = 3*ez*ex*(ex**2 + ey**2 + ez**2)
    # Third order pseudo-vector and the totally antisymmetric tensor XYZ
    p23 = ex*(ey**2 - ez**2)
    p24 = ey*(ez**2 - ex**2)
    p25 = ez*(ex**2 - ey**2)
    p26 = ex*ey*ez
    return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26]


M_t = Matrix([phi(ei.row(i)) for i in range(0, 27)]).transpose()
M_r = GramSchmidt(M_t.row(i) for i in range(0, 27))
M = zeros(0, 0)
for i in range(0, 27):
    M = M.row_insert(i, M_r[i])

# Transform velocity PDFs to moment space
m = M*fi
src.comment('Macroscopic density')
src.let(rho, m[0])
src.comment('Momentum')
src.let(jx, m[1])
src.let(jy, m[2])
src.let(jz, m[3])

src.comment('Macroscopic velocity')
src.let(ux, jx/rho)
src.let(uy, jy/rho)
src.let(uz, jz/rho)
u = Matrix([ux, uy, uz])
src.let(u_bar, ((ux)**2 + (uy)**2 + (uz)**2)**(1.0/2.0))


def eq(ei, wi):
    return wi*rho*(1 + ei.dot(u)/cs2 + (ei.dot(u)**2 - cs2*u_bar**2)/(2*cs2**2))


fi_eq = Matrix([eq(ei.row(i), d3q27_weights[i]) for i in range(0, 27)])
m_eq = M*fi_eq

src.let(mi_eq, m_eq)

src.comment('Difference to velocity equilibrium')
src.let(mi_diff, m - mi_eq)

###############################################################################
# Large eddy simulation


def Sij(alpha, beta):
    return Matrix([ei.row(i)[alpha]*ei.row(i)[beta]*mi_diff[i] for i in range(0, 27)])


src.comment('Non equilibrium stress-tensor for velocity')
src.let(Sxx, Sij(0, 0).dot(ones(1, 27)))
src.let(Syy, Sij(1, 1).dot(ones(1, 27)))
src.let(Szz, Sij(2, 2).dot(ones(1, 27)))
src.let(Sxy, Sij(0, 1).dot(ones(1, 27)))
src.let(Sxz, Sij(0, 2).dot(ones(1, 27)))
src.let(Syz, Sij(1, 2).dot(ones(1, 27)))

src.comment('Magnitude of strain rate tensor')
src.let(S_bar, ((Sxx*Sxx + Syy*Syy + Szz*Szz +
                 2.0*(Sxy*Sxy + Syz*Syz + Sxz*Sxz)))**(1.0/2.0))
src.let(ST, (1.0 / 6.0) * ((nu**2 + 18.0 * C**2 * S_bar)**(1.0/2.0) - nu))

# Collision matrix for velocities in moment space
src.comment('Modified shear viscosity')
src.let(tau_V, 1.0/(3.0*(nu + ST) + 0.5))

###############################################################################
# Relaxation matrix

s1 = 0
s4 = 1.54
s10 = 1.5
s13 = 1.83
s16 = 1.4
s17 = 1.61
s18 = s20 = 1.98
s23 = s26 = 1.74

s5 = s7 = tau_V

S_hat = sympy.diag(0, 0, 0, 0, s4, s5, s5, s7, s7, s7, s10, s10, s10,
                   s13, s13, s13, s16, s17, s18, s18, s20, s20, s20, s23, s23, s23, s26)

# S_hat = sympy.eye(27)*tau_V # Single relaxation time

M_inv = M.transpose() * (M * M.transpose())**(-1)

# Transform velocity moments to velocity
src.comment('Relax velocity')
src.let(omega, M_inv*(S_hat*mi_diff))

###############################################################################
# Temperature distribution functions

src.comment('Macroscopic temperature')
src.let(T, Ti.dot(ones(1, 7)))


def chi_eq(ei):
    b = 7.0
    return T/b*(1.0 + b/2.0*ei.dot(u))


src.comment('Temperature equilibirum distribution functions')
T_eq = Matrix([chi_eq(ei.row(i)) for i in range(0, 7)])
src.let(Ti_eq, T_eq)
src.comment('Modified relaxation time for the temperature')
src.let(tau_T, 3.0*(nuT + ST/Pr_t) + 0.5)

omegaT = Matrix([-1.0/tau_T*Ti + 1.0/tau_T*Ti_eq])

###############################################################################
# Boussinesq approximation of body force


def Fi(i):
    return (T - Tref)*gBetta*ei.row(i)


src.comment('Boussinesq approximation of body force')
src.let(Fup, Fi(5)[2])
src.let(Fdown, Fi(6)[2])
Fi = zeros(27, 1)
Fi[5] = Fup
Fi[6] = Fdown

###############################################################################
# Write distribution functions
dftmp3D = Matrix([[fi.row(i)[0] - omega.row(i)[0] + Fi.row(i)[0]]
                  for i in range(0, 27)])
Tdftmp3D = Matrix([[Ti.row(i)[0] + omegaT.row(i)[0]] for i in range(0, 7)])

src.comment('Write relaxed velocity')
for i in range(0, 27):
    src.append(f'dftmp3D({i}, x, y, z, nx, ny, nz) = {dftmp3D.row(i)[0]};')
for i in range(0, 7):
    src.append(f'Tdftmp3D({i}, x, y, z, nx, ny, nz) = {Tdftmp3D.row(i)[0]};')

src.comment('Store macroscopic values')
src.append('phy->rho = rho;')
src.append('phy->T = T;')
src.append('phy->vx = ux;')
src.append('phy->vy = uy;')
src.append('phy->vz = uz;')

src.generate(sys.argv)

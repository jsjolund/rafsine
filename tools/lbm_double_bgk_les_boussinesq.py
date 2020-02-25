"""BGK-LBM kernel generator for D3Q19 velocity and D3Q7 temperature distributios.
Implements LES for turbulence modelling and Boussinesq approximation of body force.
"""

import sys
import math
import sympy
from sympy import Matrix, diag, eye, ones, zeros, symbols, pprint
from code_printer import CodePrinter

sympy.init_printing(use_unicode=True, num_columns=220, wrap_line=False)

src = CodePrinter()

"""Kernel input constants"""
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
omega = Matrix([symbols(f'omega{i}') for i in range(0, 19)])

Ti_eq = Matrix([symbols(f'T{i}eq') for i in range(0, 7)])
Ti_neq = Matrix([symbols(f'T{i}neq') for i in range(0, 7)])
omegaT = Matrix([symbols(f'omegaT{i}') for i in range(0, 7)])

Sxx = symbols('Sxx')
Syy = symbols('Syy')
Szz = symbols('Szz')
Sxy = symbols('Sxy')
Syz = symbols('Syz')
Sxz = symbols('Sxz')
m1_1 = symbols('m1_1')
m1_9 = symbols('m1_9')
m1_11 = symbols('m1_11')
m1_13 = symbols('m1_13')
m1_14 = symbols('m1_14')
m1_15 = symbols('m1_15')
S_bar = symbols('S_bar')
ST = symbols('ST')

rho = symbols('rho')
sq_term = symbols('sq_term')
epsilon = symbols('epsilon')
jx = symbols('jx')
qx = symbols('qx')
jy = symbols('jy')
qy = symbols('qy')
jz = symbols('jz')
qz = symbols('qz')
pxx3 = symbols('pxx3')
pixx3 = symbols('pixx3')
pww = symbols('pww')
piww = symbols('piww')
pxy = symbols('pxy')
pyz = symbols('pyz')
pxz = symbols('pxz')
mx = symbols('mx')
my = symbols('my')
mz = symbols('mz')
omega_e = symbols('omega_e')
omega_xx = symbols('omega_xx')
omega_ej = symbols('omega_ej')
tau_V = symbols('tau_V')
tau_T = symbols('tau_T')

T = symbols('T')
vx = symbols('vx')
vy = symbols('vy')
vz = symbols('vz')

Fup = symbols('Fup')
Fdown = symbols('Fdown')

src.define(fi_eq, fi_neq, omega, Ti_eq,  Ti_neq, omegaT, S_bar, ST,
           T, Fup, Fdown, tau_V, tau_T,
           Matrix([Sxx, Syy, Szz, Sxy, Syz, Sxz]),
           Matrix([m1_1, m1_9, m1_11, m1_13, m1_14, m1_15]),
           Matrix([rho, sq_term, epsilon, jx, qx, jy, qy, jz, qz,
                   pxx3, pixx3, pww, piww, pxy, pyz, pxz, mx, my, mz]),
           Matrix([omega_e, omega_xx, omega_ej]),
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
# e_omegaT = Matrix([
#     1.0/4.0,
#     1.0/8.0,
#     1.0/8.0,
#     1.0/8.0,
#     1.0/8.0,
#     1.0/8.0,
#     1.0/8.0
# ])
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
    p1 = 19.0*ei.norm()**2 - 30.0
    p2 = (21.0*ei.norm()**4 - 53.0*ei.norm()**2 + 24.0)/2.0
    p3 = ei[0]
    p4 = (5.0*ei.norm()**2 - 9.0)*ei[0]
    p5 = ei[1]
    p6 = (5.0*ei.norm()**2 - 9.0)*ei[1]
    p7 = ei[2]
    p8 = (5.0*ei.norm()**2 - 9.0)*ei[2]
    p9 = 3.0*ei[0]**2 - ei.norm()**2
    p10 = (3.0*ei.norm()**2 - 5.0)*(3.0*ei[0]**2 - ei.norm()**2)
    p11 = ei[1]**2 - ei[2]**2
    p12 = (3.0*ei.norm()**2 - 5.0)*(ei[1]**2 - ei[2]**2)
    p13 = ei[0]*ei[1]
    p14 = ei[1]*ei[2]
    p15 = ei[0]*ei[2]
    p16 = (ei[1]**2 - ei[2]**2)*ei[0]
    p17 = (ei[2]**2 - ei[0]**2)*ei[1]
    p18 = (ei[0]**2 - ei[1]**2)*ei[2]
    return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18]


M = Matrix([phi(ei.row(i)) for i in range(0, 19)]).transpose()


# Transform velocity PDFs to moment space
m = M*fi
src.comment('Macroscopic density')
src.let(rho, m[0])
src.comment('Momentum')
src.let(jx, m[3])
src.let(jy, m[5])
src.let(jz, m[7])
src.comment('Macroscopic velocity')
src.let(vx, jx/rho)
src.let(vy, jy/rho)
src.let(vz, jz/rho)
V = Matrix([[vx, vy, vz]])

src.let(sq_term, -1.0*V.dot(V)/(2.0*cs2))


def phi_eq(ei, omega):
    return omega*rho*(1.0 + ei.dot(V)/cs2 + (ei.dot(V)**2)/(2.0*cs2**2) + sq_term)


src.comment('Velocity moment equilibirum distribution functions')
f_eq = Matrix([phi_eq(ei.row(i), e_omega.row(i)) for i in range(0, 19)])
src.let(fi_eq, f_eq)


# Transformation matrix for transition from energy to moment space
def chi(ei):
    p0 = ei.norm()**0
    p1 = ei[0]
    p2 = ei[1]
    p3 = ei[2]
    p4 = 6.0 - 7.0*ei.norm()**2
    p5 = 3.0*ei[0]**2 - ei.norm()**2
    p6 = ei[1]**2 - ei[2]**2
    return [p0, p1, p2, p3, p4, p5, p6]


N = Matrix([chi(ei.row(i)) for i in range(0, 7)]).transpose()

# Transform temperature PDFs to moment space
ni = N*Ti
src.comment('Macroscopic temperature')
src.let(T, ni[0])


def chi_eq(ei):
    b = 7.0
    return T/b*(1 + b/2.0*ei.dot(V))


src.comment('Temperature equilibirum distribution functions')
T_eq = Matrix([chi_eq(ei.row(i)) for i in range(0, 7)])
src.let(Ti_eq, T_eq)


# Boussinesq approximation of body force
def Fi(i):
    V = Matrix([[vx, vy, vz]])
    e = ei.row(i)
    omega = e_omega.row(i)
    # Equilibrium PDF in velocity space
    feq = rho*omega*(1 + e.dot(V)/cs2 + (e.dot(V))**2 /
                     (2.0*cs2**2) - V.dot(V)/(2.0*cs2))
    # Pressure
    p = 1.0
    return (T - Tref)*gBetta*(e - V)/p*feq[0]


src.comment('Boussinesq approximation of body force')
src.let(Fup, Fi(5)[2])
src.let(Fdown, Fi(6)[2])
Fi = zeros(19, 1)
Fi[5] = Fup
Fi[6] = Fdown

src.comment('Difference to velocity equilibrium')
src.let(fi_neq, fi - fi_eq)


def Si(i, j):
    return Matrix([ei.row(k)[i]*ei.row(k)[j]*fi_neq[k] for k in range(0, 19)])

src.comment('Non equilibrium stress-tensor for velocity')
src.let(Sxx, Si(0, 0).dot(ones(1, 19)))
src.let(Syy, Si(1, 1).dot(ones(1, 19)))
src.let(Szz, Si(2, 2).dot(ones(1, 19)))
src.let(Sxy, Si(0, 1).dot(ones(1, 19)))
src.let(Sxz, Si(0, 2).dot(ones(1, 19)))
src.let(Syz, Si(1, 2).dot(ones(1, 19)))

src.comment('Magnitude of strain rate tensor')
src.let(S_bar, (2.0*(Sxx*Sxx + Syy*Syy + Szz*Szz +
                     2.0*(Sxy*Sxy + Syz*Syz + Sxz*Sxz)))**(1.0/2.0))
src.let(ST, (1.0 / 6.0) * ( (nu * nu + 18.0 * C * C * S_bar)**(1.0/2.0) - nu))

# # Collision matrix for temperature moments
# src.comment('Modified heat diffusion')
# src.let(tau_T, 1.0/(5.0*(nuT + ST/Pr_t) + 0.5))
# tau_e = tau_v = 1.0

# tau_0 = 0.0
# tau_xx = tau_yy = tau_zz = tau_T
# tau_xy = 0.0
# tau_xz = 0.0
# tau_yz = 0.0

# Q_hat = Matrix([
#     [tau_0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, tau_xx, tau_xy, tau_xz, 0.0, 0.0, 0.0],
#     [0.0, tau_xy, tau_yy, tau_yz, 0.0, 0.0, 0.0],
#     [0.0, tau_xz, tau_yz, tau_zz, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, tau_e, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, tau_v, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tau_v],
# ])

# # Collision matrix for velocities in moment space
# src.comment('Modified shear viscosity')
# src.let(tau_V, 1.0/(3.0*(nu + ST) + 0.5))

# s1 = 1.19
# s2 = 1.4
# s4 = 1.2
# s16 = 1.98
# s9 = tau_V
# S_hat = sympy.diag(0, s1, s2, 0, s4, 0, s4, 0, s4, s9,
#                    s2, s9, s2, s9, s9, s9, s16, s16, s16)


# # Combine velocity transform and collision matrices
# Phi = M*M.transpose()
# Lambda = Phi**(-1)*S_hat

# # Transform velocity moments to velocity
# src.comment('Difference to velocity equilibrium')
# src.let(mi_neq, mi - mi_eq)
# src.comment('Relax velocity')
# src.let(omega, M.transpose()*(Lambda*mi_neq))

# # Transform energy moments back to energy
# src.comment('Difference to temperature equilibrium')
# src.let(ni_neq, ni - ni_eq)
# src.comment('Relax temperature')
# src.let(omegaT, N**(-1)*(Q_hat*ni_neq))


# # Write distribution functions
# dftmp3D = Matrix([[fi.row(i)[0] - omega.row(i)[0] + Fi.row(i)[0]]
#                   for i in range(0, 19)])
# Tdftmp3D = Matrix([[Ti.row(i)[0] - omegaT.row(i)[0]] for i in range(0, 7)])

# src.comment('Write relaxed velocity')
# for i in range(0, 19):
#     src.append(f'dftmp3D({i}, x, y, z, nx, ny, nz) = {dftmp3D.row(i)[0]};')

# src.comment('Write relaxed temperature')
# for i in range(0, 7):
#     src.append(f'Tdftmp3D({i}, x, y, z, nx, ny, nz) = {Tdftmp3D.row(i)[0]};')

if len(sys.argv) > 1:
    src.save(sys.argv[1])
else:
    print(src)

"""
A D3Q27 multiple-relaxation-time lattice Boltzmann method for turbulent flows
K. Suga, Y. Kuwata, K. Takashima, R. Chikasue
"""

import sys
import sympy
from sympy import Matrix, diag, eye, ones, zeros, symbols, pprint, GramSchmidt
from math import sqrt
from code_printer import CodePrinter

sympy.init_printing(use_unicode=True, num_columns=220, wrap_line=False)

src = CodePrinter('computeMRT')

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
# mi_eq[0] = 0
# mi_eq[3] = 0
# mi_eq[5] = 0
# mi_eq[7] = 0
mi_diff = Matrix([symbols(f'm{i}diff') for i in range(0, 27)])
# mi_diff[0] = 0
# mi_diff[3] = 0
# mi_diff[5] = 0
# mi_diff[7] = 0
omega = Matrix([symbols(f'omega{i}') for i in range(0, 27)])

ni_eq = Matrix([symbols(f'n{i}eq') for i in range(0, 7)])
ni_eq[0] = 0
ni_diff = Matrix([symbols(f'n{i}diff') for i in range(0, 7)])
ni_diff[0] = 0
omegaT = Matrix([symbols(f'omegaT{i}') for i in range(0, 7)])

Sxx = symbols('Sxx')
Syy = symbols('Syy')
Szz = symbols('Szz')
Sxy = symbols('Sxy')
Syz = symbols('Syz')
Sxz = symbols('Sxz')
S_bar = symbols('S_bar')
ST = symbols('ST')

rho = symbols('rho')
en = symbols('en')
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
ux = symbols('ux')
uy = symbols('uy')
uz = symbols('uz')
u_bar = symbols('u_bar')

Fup = symbols('Fup')
Fdown = symbols('Fdown')

src.define(mi_eq, mi_diff, omega, ni_eq,  ni_diff, omegaT, S_bar, ST,
           T, Fup, Fdown, tau_V, tau_T,
           Matrix([Sxx, Syy, Szz, Sxy, Syz, Sxz]),
           Matrix([rho, en, epsilon, jx, qx, jy, qy, jz, qz,
                   pxx3, pixx3, pww, piww, pxy, pyz, pxz, mx, my, mz]),
           Matrix([omega_e, omega_xx, omega_ej]),
           Matrix([ux, uy, uz]))

"""Kernel generator"""
# LBM velocity vectors for D3Q27
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

# Density weighting factors for D3Q19 velocity PDFs
e_omega = Matrix([
    8.0/27.0,
    2.0/27.0,
    2.0/27.0,
    2.0/27.0,
    2.0/27.0,
    2.0/27.0,
    2.0/27.0,
    1.0/54.0,
    1.0/54.0,
    1.0/54.0,
    1.0/54.0,
    1.0/54.0,
    1.0/54.0,
    1.0/54.0,
    1.0/54.0,
    1.0/54.0,
    1.0/54.0,
    1.0/54.0,
    1.0/54.0,
    1.0/216,
    1.0/216,
    1.0/216,
    1.0/216,
    1.0/216,
    1.0/216,
    1.0/216,
    1.0/216
])


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
src.let(u_bar, ((jx/rho)**2 + (jy/rho)**2 + (jz/rho)**2)**1.0/2.0)


def eq(ei, wi):
    return wi*rho*(1 + ei.dot(u)/cs2 + (ei.dot(u)**2 - cs2*u_bar**2)/(2*cs2**2))


fi_eq = Matrix([eq(ei.row(i), e_omega[i]) for i in range(0, 27)])
m_eq = M*fi_eq

src.let(mi_eq, m_eq)

src.comment('Difference to velocity equilibrium')
src.let(mi_diff, m - mi_eq)


s4 = 1.54
s10 = 1.5
s13 = 1.83
s16 = 1.4
s17 = 1.61
s18 = s20 = 1.98
s23 = s26 = 1.74

s5 = s7 = 8*(s10 - 2)/(s10 - 8)

S_hat = sympy.diag(0, 0, 0, 0, s4, s5, s5, s7, s7, s7, s10, s10, s10, s13, s13,
                   s13, s16, s17, s18, s18, s20, s20, s20, s23, s23, s23, s26)

# src.comment('Non equilibrium stress-tensor for velocity')
# src.let(Sxx, Sij(0, 0).dot(ones(1, 19)))
# src.let(Syy, Sij(1, 1).dot(ones(1, 19)))
# src.let(Szz, Sij(2, 2).dot(ones(1, 19)))
# src.let(Sxy, Sij(0, 1).dot(ones(1, 19)))
# src.let(Sxz, Sij(0, 2).dot(ones(1, 19)))
# src.let(Syz, Sij(1, 2).dot(ones(1, 19)))

# src.comment('LES strain rate tensor')
# src.let(Sxx, -1.0/(38.0*rho_0*dt)*(1.0*s1*m[1]+19.0*s9*m[9]))
# src.let(Syy, -1.0/(76.0*rho_0*dt)*(2.0*s1*m[1]-19.0*s9*(m[9]-3.0*m[11])))
# src.let(Szz, -1.0/(76.0*rho_0*dt)*(2.0*s1*m[1]-19.0*s9*(m[9]+3.0*m[11])))
# src.let(Sxy, -3.0*s9/(2.0*rho_0*dt)*m[13])
# src.let(Syz, -3.0*s9/(2.0*rho_0*dt)*m[14])
# src.let(Sxz, -3.0*s9/(2.0*rho_0*dt)*m[15])

# src.comment('Magnitude of strain rate tensor')
# src.let(S_bar, (2.0*(Sxx*Sxx + Syy*Syy + Szz*Szz +
#                      2.0*(Sxy*Sxy + Syz*Syz + Sxz*Sxz)))**(1.0/2.0))
# src.comment('Filtered strain rate')
# src.let(ST, (C*dfw)**2*S_bar)


# # Transformation matrix for transition from energy to moment space
# def chi(ei):
#     p0 = ei.norm()**0
#     p1 = ei[0]
#     p2 = ei[1]
#     p3 = ei[2]
#     p4 = 6.0 - 7.0*ei.norm()**2
#     p5 = 3.0*ei[0]**2 - ei.norm()**2
#     p6 = ei[1]**2 - ei[2]**2
#     return [p0, p1, p2, p3, p4, p5, p6]


# N = Matrix([chi(ei.row(i)) for i in range(0, 7)]).transpose()

# # Transform temperature PDFs to moment space
# ni = N*Ti
# src.comment('Macroscopic temperature')
# src.let(T, ni[0])

# # Temperature moment equilibrium PDFs
# a = 0.75
# n_eq0 = T
# n_eq1 = ux*T
# n_eq2 = uy*T
# n_eq3 = uz*T
# n_eq4 = a*T
# n_eq5 = 0.0
# n_eq6 = 0.0
# n_eq = Matrix([n_eq0, n_eq1, n_eq2, n_eq3, n_eq4, n_eq5, n_eq6])

# src.comment('Temperature moment equilibirum distribution functions')
# src.let(ni_eq, n_eq)


# # # Boussinesq approximation of body force
# # def Fi(i):
# #     V = Matrix([[ux, uy, uz]])
# #     e = ei.row(i)
# #     omega = e_omega.row(i)
# #     # Equilibrium PDF in velocity space
# #     feq = rho*omega*(1 + e.dot(V)/cs2 + (e.dot(V))**2 /
# #                      (2.0*cs2**2) - V.dot(V)/(2.0*cs2))
# #     # Pressure
# #     p = 1.0
# #     return (T - Tref)*gBetta*(e - V)/p*feq[0]


# # src.comment('Boussinesq approximation of body force')
# # src.let(Fup, Fi(5)[2])
# # src.let(Fdown, Fi(6)[2])
# # Fi = zeros(19, 1)
# # Fi[5] = Fup
# # Fi[6] = Fdown

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
# s9 = tau_V
# S_hat = sympy.diag(0, 0, 0, 0, s4, s5, s5, s7, s7, s7, s10, s10, s10,
#                    s13, s13, s13, s16, s17, s18, s18, s20, s20, s20, s23, s23, s23, s26)


# # Transform velocity moments to velocity
# src.comment('Relax velocity')
# src.let(omega, M**(-1)*(S_hat*mi_diff))

# # Transform energy moments back to energy
# src.comment('Difference to temperature equilibrium')
# src.let(ni_diff, ni - ni_eq)
# src.comment('Relax temperature')
# src.let(omegaT, N**(-1)*(Q_hat*ni_diff))


# # Write distribution functions
# # dftmp3D = Matrix([[fi.row(i)[0] - omega.row(i)[0] + Fi.row(i)[0]] for i in range(0, 19)])
# dftmp3D = Matrix([[fi.row(i)[0] - omega.row(i)[0]] for i in range(0, 19)])
# Tdftmp3D = Matrix([[Ti.row(i)[0] - omegaT.row(i)[0]] for i in range(0, 7)])

# src.comment('Write relaxed velocity')
# for i in range(0, 19):
#     src.append(f'dftmp3D({i}, x, y, z, nx, ny, nz) = {dftmp3D.row(i)[0]};')

# src.comment('Write relaxed temperature')
# for i in range(0, 7):
#     src.append(f'Tdftmp3D({i}, x, y, z, nx, ny, nz) = {Tdftmp3D.row(i)[0]};')

# src.comment('Store macroscopic values')
# src.append('phy->rho = rho;')
# src.append('phy->T = T;')
# src.append('phy->ux = ux;')
# src.append('phy->uy = uy;')
# src.append('phy->uz = uz;')

# src.include("CudaUtils.hpp")
# src.include("PhysicalQuantity.hpp")

src.generate(sys.argv)

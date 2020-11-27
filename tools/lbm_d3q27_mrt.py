"""
Rectangular lattice Boltzmann method using multiple relaxation time collision operator in two and three dimensions
"""

import sys
import sympy
from sympy import Matrix, diag, eye, ones, zeros, symbols, pprint
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

# # Density weighting factors for D3Q19 velocity PDFs
# e_omega = Matrix([
#     1.0/3.0,
#     1.0/18.0,
#     1.0/18.0,
#     1.0/18.0,
#     1.0/18.0,
#     1.0/18.0,
#     1.0/18.0,
#     1.0/36.0,
#     1.0/36.0,
#     1.0/36.0,
#     1.0/36.0,
#     1.0/36.0,
#     1.0/36.0,
#     1.0/36.0,
#     1.0/36.0,
#     1.0/36.0,
#     1.0/36.0,
#     1.0/36.0,
#     1.0/36.0
# ])


def phi(ei):
    # Transformation matrix for transition from velocity to moment space
    cx = ei[0]
    cy = ei[1]
    cz = ei[2]
    c0 = ei.norm()**0
    p0 = c0
    p1 = cx
    p2 = cy
    p3 = cz
    p4 = 3*cx**2 - 2*dx**2*c0
    p5 = 3*cy**2 - 2*dy**2*c0
    p6 = 3*cz**2 - 2*dz**2*c0
    p7 = cx*cy
    p8 = cx*cz
    p9 = cy*cz
    p10 = 3*cx*cy**2 - 2*dy**2*cx
    p11 = 3*cx*cz**2 - 2*dz**2*cx
    p12 = 3*cy*cx**2 - 2*dx**2*cy
    p13 = 3*cy*cz**2 - 2*dz**2*cy
    p14 = 3*cz*cx**2 - 2*dx**2*cz
    p15 = 3*cz*cy**2 - 2*dy**2*cz
    p16 = cx*cy*cz
    p17 = 9*cx**2*cy**2 - 6*dx**2*cy**2 - 6*dy**2*cy**2 + 4*dx**2*dy**2
    p18 = 9*cx**2*cz**2 - 6*dx**2*cz**2 - 6*dz**2*cx**2 + 4*dx**2*dz**2
    p19 = 9*cy**2*cz**2 - 6*dy**2*cz**2 - 6*dz**2*cy**2 + 4*dy**2*dz**2
    p20 = 9*cx**2*cy*cz - 6*dx**2*cy*cz
    p21 = 9*cy**2*cx*cz - 6*dy**2*cx*cz
    p22 = 9*cz**2*cx*cy - 6*dz**2*cx*cy
    p23 = 9*cx**2*cy**2*cz - 6*dx**2*cy**2*cz - 6*dy**2*cx**2*cz + 4*dx**2*dy**2*cz
    p24 = 9*cx**2*cz**2*cx - 6*dx**2*cz**2*cy - 6*dz**2*cx**2*cy + 4*dx**2*dz**2*cy
    p25 = 9*cy**2*cz**2*cx - 6*dy**2*cz**2*cx - 6*dz**2*cy**2*cx + 4*dy**2*dz**2*cx
    p26 = 27*cx**2*cy**2*cz**2 - 8*dx**2*dy**2*dz**2*c0 - 18 * \
        (dx**2*cy**2*cz**2 + dy**2*cx**2*cz**2 + dz**2*cx**2*cy**2) + \
        12*(dx**2*dy**2*cz**2 + dx**2*dz**2*cy**2 + dy**2*dz**2*cx**2)
    return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26]


M = Matrix([phi(ei.row(i)) for i in range(0, 27)]).transpose()
# pprint(M)

# Transform velocity PDFs to moment space
m = M*fi
src.comment('Macroscopic density')
src.let(rho, m[0])
# src.comment('Part of kinetic energy independent of density')
# src.let(en, m[1])
# src.comment('Kinetic energy square')
# src.let(epsilon, m[2])
src.comment('Momentum')
src.let(jx, m[1])
src.let(jy, m[2])
src.let(jz, m[3])
# src.comment('Energy flux independent of mass flux')
# src.let(qx, m[4])
# src.let(qy, m[6])
# src.let(qz, m[8])
# src.comment('Symmetric traceless viscous stress tensor')
# src.let(pxx3, m[9])
# src.let(pww, m[11])
# src.let(pxy, m[13])
# src.let(pyz, m[14])
# src.let(pxz, m[15])
# src.comment('Fourth order moments')
# src.let(pixx3, m[10])
# src.let(piww, m[12])
# src.comment('Antisymmetric third-order moment')
# src.let(mx, m[16])
# src.let(my, m[17])
# src.let(mz, m[18])

# mi = Matrix([rho, en, epsilon, jx, qx, jy, qy, jz, qz, pxx3,
#              pixx3, pww, piww, pxy, pyz, pxz, mx, my, mz])

src.comment('Model stability constants')
src.let(omega_e, 0)
src.let(omega_xx, 0)
src.let(omega_ej, -475.0/63.0)

src.comment('Macroscopic velocity')
src.let(ux, jx/rho)
src.let(uy, jy/rho)
src.let(uz, jz/rho)

src.comment('Velocity moment equilibirum distribution functions')
h = dx
gamma2 = h**4/9
gamma3 = h**4/3
gamma4 = h**2/3
gamma5 = h**4/9
gamma6 = h**6/27
gamma7 = h**4/9

m_eq0 = rho
m_eq1 = rho*ux
m_eq2 = rho*uy
m_eq3 = rho*uz
m_eq4 = rho*(3*cs2 - 2*dx**2) + 3*rho*ux*ux
m_eq5 = rho*(3*cs2 - 2*dy**2) + 3*rho*uy*uy
m_eq6 = rho*(3*cs2 - 2*dz**2) + 3*rho*uz*uz
m_eq7 = rho*ux*uy
m_eq8 = rho*ux*uz
m_eq9 = rho*uy*uz
m_eq10 = rho*ux*(3*cs2 - 2*dy**2)
m_eq11 = rho*ux*(3*cs2 - 2*dz**2)
m_eq12 = rho*uy*(3*cs2 - 2*dx**2)
m_eq13 = rho*uy*(3*cs2 - 2*dz**2)
m_eq14 = rho*uz*(3*cs2 - 2*dx**2)
m_eq15 = rho*uz*(3*cs2 - 2*dy**2)
m_eq16 = 0
m_eq17 = rho*(9*gamma2 - 6*cs2*(dx**2 + dy**2) + 4*dx**2*dy**2) + \
    rho*ux*ux*(9*gamma3 - 6*dy**2) + rho*uy*uy*(9*gamma3 - 6*dx**2)
m_eq18 = rho*(9*gamma2 - 6*cs2*(dx**2 + dz**2) + 4*dx**2*dz**2) + \
    rho*ux*ux*(9*gamma3 - 6*dz**2) + rho*uz*uz*(9*gamma3 - 6*dx**2)
m_eq19 = rho*(9*gamma2 - 6*cs2*(dy**2 + dz**2) + 4*dy**2*dz**2) + \
    rho*uy*uy*(9*gamma3 - 6*dz**2) + rho*uz*uz*(9*gamma3 - 6*dy**2)
m_eq20 = rho*uy*uz*(9*gamma4 - 6*dx**2)
m_eq21 = rho*uy*uz*(9*gamma4 - 6*dy**2)
m_eq22 = rho*ux*uy*(9*gamma4 - 6*dz**2)
m_eq23 = rho*uz*(9*gamma5 - 6*cs2*(dx**2 + dy**2) + 4*dx**2*dy**2)
m_eq24 = rho*uy*(9*gamma5 - 6*cs2*(dx**2 + dz**2) + 4*dx**2*dz**2)
m_eq25 = rho*ux*(9*gamma5 - 6*cs2*(dy**2 + dz**2) + 4*dy**2*dz**2)
m_eq26 = rho*(27*gamma6 - 18*gamma2*(dx**2 + dy**2 + dz**2) - 8*dx**2*dy**2*dz**2 + 12*cs2*(dx**2*dy**2 + dx**2*dz**2 + dy**2*dz**2)) \
    + rho*ux*ux*(27*gamma7 - 18*gamma3*(dy**2 + dz**2) + 12*dy**2*dz**2) \
    + rho*uy*uy*(27*gamma7 - 18*gamma3*(dx**2 + dz**2) + 12*dx**2*dz**2) \
    + rho*uz*uz*(27*gamma7 - 18*gamma3*(dx**2 + dy**2) + 12*dx**2*dy**2)


m_eq = Matrix([m_eq0, m_eq1, m_eq2, m_eq3, m_eq4, m_eq5, m_eq6, m_eq7, m_eq8,
               m_eq9, m_eq10, m_eq11, m_eq12, m_eq13, m_eq14, m_eq15, m_eq16, m_eq17, m_eq18,
               m_eq19, m_eq20, m_eq21, m_eq22, m_eq23, m_eq24, m_eq25, m_eq26])
src.let(mi_eq, m_eq)

src.comment('Difference to velocity equilibrium')
src.let(mi_diff, m - mi_eq)

# S_hat = sympy.diag(0, s1, s2, 0, s4, 0, s4, 0, s4, s9, s2, s9, s2, s9, s9, s9, s16, s16, s16)

s1 = 1.19
s2 = 1.4
s4 = 1.2
s16 = 1.98
s9 = 1.0/(3.0*nu + 0.5)

# A D3Q27 multiple-relaxation-time lattice Boltzmann method for turbulent flows
s4 = 1.54
s10 = 1.5
s13 = 1.83
s16 = 1.4
s17 = 1.61
s18 = s20 = 1.98
s23 = s26 = 1.74

src.comment('Non equilibrium stress-tensor for velocity')
src.let(Sxx, Sij(0, 0).dot(ones(1, 19)))
src.let(Syy, Sij(1, 1).dot(ones(1, 19)))
src.let(Szz, Sij(2, 2).dot(ones(1, 19)))
src.let(Sxy, Sij(0, 1).dot(ones(1, 19)))
src.let(Sxz, Sij(0, 2).dot(ones(1, 19)))
src.let(Syz, Sij(1, 2).dot(ones(1, 19)))

src.comment('LES strain rate tensor')
src.let(Sxx, -1.0/(38.0*rho_0*dt)*(1.0*s1*m[1]+19.0*s9*m[9]))
src.let(Syy, -1.0/(76.0*rho_0*dt)*(2.0*s1*m[1]-19.0*s9*(m[9]-3.0*m[11])))
src.let(Szz, -1.0/(76.0*rho_0*dt)*(2.0*s1*m[1]-19.0*s9*(m[9]+3.0*m[11])))
src.let(Sxy, -3.0*s9/(2.0*rho_0*dt)*m[13])
src.let(Syz, -3.0*s9/(2.0*rho_0*dt)*m[14])
src.let(Sxz, -3.0*s9/(2.0*rho_0*dt)*m[15])

src.comment('Magnitude of strain rate tensor')
src.let(S_bar, (2.0*(Sxx*Sxx + Syy*Syy + Szz*Szz +
                     2.0*(Sxy*Sxy + Syz*Syz + Sxz*Sxz)))**(1.0/2.0))
src.comment('Filtered strain rate')
src.let(ST, (C*dfw)**2*S_bar)


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

# Temperature moment equilibrium PDFs
a = 0.75
n_eq0 = T
n_eq1 = ux*T
n_eq2 = uy*T
n_eq3 = uz*T
n_eq4 = a*T
n_eq5 = 0.0
n_eq6 = 0.0
n_eq = Matrix([n_eq0, n_eq1, n_eq2, n_eq3, n_eq4, n_eq5, n_eq6])

src.comment('Temperature moment equilibirum distribution functions')
src.let(ni_eq, n_eq)


# # Boussinesq approximation of body force
# def Fi(i):
#     V = Matrix([[ux, uy, uz]])
#     e = ei.row(i)
#     omega = e_omega.row(i)
#     # Equilibrium PDF in velocity space
#     feq = rho*omega*(1 + e.dot(V)/cs2 + (e.dot(V))**2 /
#                      (2.0*cs2**2) - V.dot(V)/(2.0*cs2))
#     # Pressure
#     p = 1.0
#     return (T - Tref)*gBetta*(e - V)/p*feq[0]


# src.comment('Boussinesq approximation of body force')
# src.let(Fup, Fi(5)[2])
# src.let(Fdown, Fi(6)[2])
# Fi = zeros(19, 1)
# Fi[5] = Fup
# Fi[6] = Fdown

# Collision matrix for temperature moments
src.comment('Modified heat diffusion')
src.let(tau_T, 1.0/(5.0*(nuT + ST/Pr_t) + 0.5))
tau_e = tau_v = 1.0

tau_0 = 0.0
tau_xx = tau_yy = tau_zz = tau_T
tau_xy = 0.0
tau_xz = 0.0
tau_yz = 0.0

Q_hat = Matrix([
    [tau_0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, tau_xx, tau_xy, tau_xz, 0.0, 0.0, 0.0],
    [0.0, tau_xy, tau_yy, tau_yz, 0.0, 0.0, 0.0],
    [0.0, tau_xz, tau_yz, tau_zz, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, tau_e, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, tau_v, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tau_v],
])

# Collision matrix for velocities in moment space
src.comment('Modified shear viscosity')
src.let(tau_V, 1.0/(3.0*(nu + ST) + 0.5))
s9 = tau_V
S_hat = sympy.diag(0, 0, 0, 0, s4, s5, s5, s7, s7, s7, s10, s10, s10,
                   s13, s13, s13, s16, s17, s18, s18, s20, s20, s20, s23, s23, s23, s26)


# Transform velocity moments to velocity
src.comment('Relax velocity')
src.let(omega, M**(-1)*(S_hat*mi_diff))

# Transform energy moments back to energy
src.comment('Difference to temperature equilibrium')
src.let(ni_diff, ni - ni_eq)
src.comment('Relax temperature')
src.let(omegaT, N**(-1)*(Q_hat*ni_diff))


# Write distribution functions
# dftmp3D = Matrix([[fi.row(i)[0] - omega.row(i)[0] + Fi.row(i)[0]] for i in range(0, 19)])
dftmp3D = Matrix([[fi.row(i)[0] - omega.row(i)[0]] for i in range(0, 19)])
Tdftmp3D = Matrix([[Ti.row(i)[0] - omegaT.row(i)[0]] for i in range(0, 7)])

src.comment('Write relaxed velocity')
for i in range(0, 19):
    src.append(f'dftmp3D({i}, x, y, z, nx, ny, nz) = {dftmp3D.row(i)[0]};')

src.comment('Write relaxed temperature')
for i in range(0, 7):
    src.append(f'Tdftmp3D({i}, x, y, z, nx, ny, nz) = {Tdftmp3D.row(i)[0]};')

src.comment('Store macroscopic values')
src.append('phy->rho = rho;')
src.append('phy->T = T;')
src.append('phy->ux = ux;')
src.append('phy->uy = uy;')
src.append('phy->uz = uz;')

src.include("CudaUtils.hpp")
src.include("PhysicalQuantity.hpp")

src.generate(sys.argv)

"""MRT-LBM kernel generator for D3Q19 velocity and D3Q7 temperature distributios.
Implements LES for turbulence modelling and Boussinesq approximation of body force.

2D AND 3D VERIFICATION AND VALIDATION OF THE LATTICE BOLTZMANN METHOD
https://publications.polymtl.ca/1927/1/2015_MatteoPortinari.pdf

Lattice Boltzmann Method Simulation of 3-D Natural Convection with Double MRT Model
https://arxiv.org/pdf/1511.04633.pdf

Multiple-relaxation-time Lattice Boltzmann Models in 3D
https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20020075050.pdf

Multiple-relaxation-time lattice Boltzmann model for the convection and anisotropic diffusion equation
https://www.researchgate.net/publication/222659771_Multiple-relaxation-time_lattice_Boltzmann_model_for_the_convection_and_anisotropic_diffusion_equation
"""

import sys
from pathlib import Path

import math
import sympy
from sympy import Matrix, diag, eye, ones, zeros, symbols, pprint
from sympy.codegen.ast import Assignment
from sympy.printing.ccode import C99CodePrinter

sympy.init_printing(use_unicode=True, num_columns=220, wrap_line=False)


class MyCodePrinter(C99CodePrinter):
    """Code printer for CUDA LBM kernel generator"""

    def __init__(self):
        super().__init__()
        self.rows = []

    def define(self, *var, type='real'):
        """Define a variable"""
        for v in var:
            if isinstance(v, Matrix):
                self.rows += [
                    f'{type} {", ".join([str(v.row(i)[0]) for i in range(0, v.shape[0])])};']
            else:
                self.rows += [f'{type} {v};']

    def comment(self, expr):
        """Append a comment"""
        self.rows += [f'\n// {expr}']

    def append(self, expr):
        """Append an expression from string"""
        self.rows += [expr]

    def let(self, var, expr):
        """Assign a variable"""
        if isinstance(var, Matrix):
            for i in range(0, var.shape[0]):
                self.rows += [self.doprint(Assignment(var.row(i)
                                                      [0], expr.row(i)[0]))]
        else:
            self.rows += [self.doprint(Assignment(var, expr))]

    def __repr__(self):
        return '\n'.join(self.rows)


src = MyCodePrinter()

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
mi_eq = Matrix([symbols(f'm{i}eq') for i in range(0, 19)])
mi_neq = Matrix([symbols(f'm{i}neq') for i in range(0, 19)])
omega = Matrix([symbols(f'omega{i}') for i in range(0, 19)])

ni_eq = Matrix([symbols(f'n{i}eq') for i in range(0, 7)])
ni_neq = Matrix([symbols(f'n{i}neq') for i in range(0, 7)])
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
vx = symbols('vx')
vy = symbols('vy')
vz = symbols('vz')

Fup = symbols('Fup')
Fdown = symbols('Fdown')

src.define(mi_eq, mi_neq, omega, ni_eq,  ni_neq, omegaT, S_bar, ST,
           T, Fup, Fdown, tau_V, tau_T,
           Matrix([Sxx, Syy, Szz, Sxy, Syz, Sxz]),
           Matrix([m1_1, m1_9, m1_11, m1_13, m1_14, m1_15]),
           Matrix([rho, en, epsilon, jx, qx, jy, qy, jz, qz,
                   pxx3, pixx3, pww, piww, pxy, pyz, pxz, mx, my, mz]),
           Matrix([omega_e, omega_xx, omega_ej]),
           Matrix([vx, vy, vz]))

"""Kernel generator"""
# LBM velocity vectors for D3Q19 (and D3Q7)
ei = Matrix([
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 0, 1],
    [-1, 0, -1],
    [1, 0, -1],
    [-1, 0, 1],
    [0, 1, 1],
    [0, -1, -1],
    [0, 1, -1],
    [0, -1, 1]
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
    1.0/4.0,
    1.0/8.0,
    1.0/8.0,
    1.0/8.0,
    1.0/8.0,
    1.0/8.0,
    1.0/8.0
])
# e_omegaT = Matrix([
#     0.0,
#     1.0/6.0,
#     1.0/6.0,
#     1.0/6.0,
#     1.0/6.0,
#     1.0/6.0,
#     1.0/6.0
# ])


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
src.comment('Part of kinetic energy independent of density')
src.let(en, m[1])
src.comment('Kinetic energy square')
src.let(epsilon, m[2])
src.comment('Momentum')
src.let(jx, m[3])
src.let(jy, m[5])
src.let(jz, m[7])
src.comment('Energy flux independent of mass flux')
src.let(qx, m[4])
src.let(qy, m[6])
src.let(qz, m[8])
src.comment('Symmetric traceless viscous stress tensor')
src.let(pxx3, m[9])
src.let(pww, m[11])
src.let(pxy, m[13])
src.let(pyz, m[14])
src.let(pxz, m[15])
src.comment('Fourth order moments')
src.let(pixx3, m[10])
src.let(piww, m[12])
src.comment('Antisymmetric third-order moment')
src.let(mx, m[16])
src.let(my, m[17])
src.let(mz, m[18])

mi = Matrix([rho, en, epsilon, jx, qx, jy, qy, jz, qz, pxx3,
             pixx3, pww, piww, pxy, pyz, pxz, mx, my, mz])

src.comment('Model stability constants')
src.let(omega_e, 0)
src.let(omega_xx, 0)
src.let(omega_ej, -475.0/63.0)

src.comment('Macroscopic velocity')
src.let(vx, jx/rho)
src.let(vy, jy/rho)
src.let(vz, jz/rho)

src.comment('Velocity moment equilibirum distribution functions')
m_eq0 = rho
m_eq1 = -11.0*rho + 19.0/rho_0*(jx*jx + jy*jy + jz*jz)
m_eq2 = omega_e*rho + omega_ej/rho_0*(jx*jx + jy*jy + jz*jz)
m_eq3 = jx
m_eq4 = -2.0/3.0*jx
m_eq5 = jy
m_eq6 = -2.0/3.0*jy
m_eq7 = jz
m_eq8 = -2.0/3.0*jz
m_eq9 = 1.0/rho_0*(2.0*jx*jx - jy*jy - jz*jz)
m_eq10 = omega_xx*m_eq9
m_eq11 = 1.0/rho_0*(jy*jy - jz*jz)
m_eq12 = omega_xx*m_eq11
m_eq13 = 1.0/rho_0*jx*jy
m_eq14 = 1.0/rho_0*jy*jz
m_eq15 = 1.0/rho_0*jx*jz
m_eq16 = 0
m_eq17 = 0
m_eq18 = 0
m_eq = Matrix([m_eq0, m_eq1, m_eq2, m_eq3, m_eq4, m_eq5, m_eq6, m_eq7, m_eq8,
               m_eq9, m_eq10, m_eq11, m_eq12, m_eq13, m_eq14, m_eq15, m_eq16, m_eq17, m_eq18])
src.let(mi_eq, m_eq)

src.comment('LES strain rate tensor')
src.let(m1_1, 38.0/3.0*(jx + jy + jz))
src.let(m1_9, -2.0/3.0*(2.0*jx - jy - jz))
src.let(m1_11, -2.0/3.0*(jy - jz))
src.let(m1_13, -1.0/3.0*(jx + jy))
src.let(m1_14, -1.0/3.0*(jz + jy))
src.let(m1_15, -1.0/3.0*(jx + jz))

src.let(Sxx, -m1_1/(38.0*rho_0*dt) - m1_9/(2.0*rho_0*dt))
src.let(Syy, -m1_1/(38.0*rho_0*dt) + m1_9 /
        (4.0*rho_0*dt) - 3.0*m1_11/(4.0*rho_0*dt))
src.let(Szz, -m1_1/(38.0*rho_0*dt) + m1_9 /
        (4.0*rho_0*dt) + 3.0*m1_11/(4.0*rho_0*dt))
src.let(Sxy, -3.0*m1_13/(2.0*rho_0*dt))
src.let(Syz, -3.0*m1_14/(2.0*rho_0*dt))
src.let(Sxz, -3.0*m1_15/(2.0*rho_0*dt))

src.comment('Magnitude of strain rate tensor')
src.let(S_bar, (2.0*(Sxx*Sxx + Syy*Syy + Szz*Szz +
                     Sxy*Sxy + Syz*Syz + Sxz*Sxz))**(1.0/2.0))

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
n_eq1 = vx*T
n_eq2 = vy*T
n_eq3 = vz*T
n_eq4 = a*T
n_eq5 = 0.0
n_eq6 = 0.0
n_eq = Matrix([n_eq0, n_eq1, n_eq2, n_eq3, n_eq4, n_eq5, n_eq6])

src.comment('Temperature moment equilibirum distribution functions')
src.let(ni_eq, n_eq)


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

s1 = 1.19
s2 = 1.4
s4 = 1.2
s16 = 1.98
s9 = tau_V
S_hat = sympy.diag(0, s1, s2, 0, s4, 0, s4, 0, s4, s9,
                   s2, s9, s2, s9, s9, s9, s16, s16, s16)


# Combine velocity transform and collision matrices
Phi = M*M.transpose()
Lambda = Phi**(-1)*S_hat

# Transform velocity moments to velocity
src.comment('Difference to velocity equilibrium')
src.let(mi_neq, mi - mi_eq)
src.comment('Relax velocity')
src.let(omega, M.transpose()*(Lambda*mi_neq))

# Transform energy moments back to energy
src.comment('Difference to temperature equilibrium')
src.let(ni_neq, ni - ni_eq)
src.comment('Relax temperature')
src.let(omegaT, N**(-1)*(Q_hat*ni_neq))


# Write distribution functions
dftmp3D = Matrix([[fi.row(i)[0] - omega.row(i)[0] + Fi.row(i)[0]]
                  for i in range(0, 19)])
Tdftmp3D = Matrix([[Ti.row(i)[0] - omegaT.row(i)[0]] for i in range(0, 7)])

src.comment('Write relaxed velocity')
for i in range(0, 19):
    src.append(f'dftmp3D({i}, x, y, z, nx, ny, nz) = {dftmp3D.row(i)[0]};')

src.comment('Write relaxed temperature')
for i in range(0, 7):
    src.append(f'Tdftmp3D({i}, x, y, z, nx, ny, nz) = {Tdftmp3D.row(i)[0]};')

if len(sys.argv) > 1:
    try:
        output_filepath = Path(sys.argv[1])
        if output_filepath.is_dir():
            raise FileNotFoundError('Error: Output path is a directory')
        with open(output_filepath, 'w') as file_to_write:
            file_to_write.write(str(src))
            print(f'Wrote to {output_filepath}')
    except Exception as e:
        print(f'{e}')
else:
    print(src)
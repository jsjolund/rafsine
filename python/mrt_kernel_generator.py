import math
import sympy
from sympy import Matrix, diag, symbols, pprint
from sympy.codegen.ast import Assignment
from sympy.printing.ccode import C99CodePrinter

sympy.init_printing(use_unicode=True, num_columns=220, wrap_line=False)

class MyCodePrinter(C99CodePrinter):
    def __init__(self):
        super().__init__()
        self.lines = []

    def define(self, *var, type='real'):
        for v in var:
            if isinstance(v, Matrix):
                self.lines += [
                    f'{type} {", ".join([str(v.row(i)[0]) for i in range(0, v.shape[0])])};']
            else:
                self.lines += [f'{type} {v};']

    def append(self, expr):
        self.lines += [expr]

    def let(self, var, expr):
        if isinstance(var, Matrix):
            for i in range(0, var.shape[0]):
                self.lines += [self.doprint(Assignment(var.row(i)[0], expr.row(i)[0]))]
        else:
            self.lines += [self.doprint(Assignment(var, expr))]
    
    def __repr__(self):
        return "\n".join(self.lines)

code = MyCodePrinter()

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
# Velocity PDFs
fi = Matrix([symbols(f'f{i}') for i in range(0, 19)])
# Temperature PDFs
Ti = Matrix([symbols(f'T{i}') for i in range(0, 7)])

# Mean density in system
rho_0 = 1
dx = 1
dt = 1

# Temporary variables
mi_eq = Matrix([symbols(f'm{i}eq') for i in range(0, 19)])
mi_neq = Matrix([symbols(f'm{i}neq') for i in range(0, 19)])
mi_diff = Matrix([symbols(f'm{i}diff') for i in range(0, 19)])
omega = Matrix([symbols(f'omega{i}') for i in range(0, 19)])
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
nu_t = symbols('nu_t')

rho = symbols('rho')
en = symbols('en')
epsilon = symbols('epsilon')
jx = symbols('jx')
qx = symbols('qx')
jy = symbols('jy')
qy = symbols('qy')
jz = symbols('jz')
qz = symbols('qz')
pxx = symbols('pxx')
pixx = symbols('pixx')
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

T = symbols('T')
vx = symbols('vx')
vy = symbols('vy')
vz = symbols('vz')
V = Matrix([vx, vy, vz])

code.define(T, mi_eq, mi_neq, mi_diff, omega, Matrix([Sxx, Syy, Szz, Sxy, Syz, Sxz]), Matrix([m1_1, m1_9, m1_11, m1_13, m1_14, m1_15]), S_bar, nu_t, Matrix([rho, en, epsilon, jx, qx, jy, qy, jz, qz, pxx, pixx, pww, piww, pxy, pyz, pxz, mx, my, mz]), Matrix([omega_e, omega_xx, omega_ej]), V)

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
    1/3,
    1/18,
    1/18,
    1/18,
    1/18,
    1/18,
    1/18,
    1/36,
    1/36,
    1/36,
    1/36,
    1/36,
    1/36,
    1/36,
    1/36,
    1/36,
    1/36,
    1/36,
    1/36
])

# Density weighting factors for D3Q7 energy PDFs
e_omegaT = Matrix([
    1/4,
    1/8,
    1/8,
    1/8,
    1/8,
    1/8,
    1/8
])

code.let(T, (sympy.ones(1, 7)*Ti)[0])

# def feq(ei):
#     cs2 = 1/3  # Speed of sound squared
#     return rho*e_omega*(1 + ei.dot(V)/cs2 + (ei.dot(V))**2/(2*cs2**2) - V.dot(V)/(2*cs2))

# # Equilibrium PDFs in velocity space
# code.let(fi_eq, Matrix([feq(ei.row(i)) for i in range(0, 19)]))

# Transformation matrix for transition from velocity to moment space
def phi(ei):
    p0 = ei.norm()**0
    p1 = 19*ei.norm()**2 - 30
    p2 = (21*ei.norm()**4 - 53*ei.norm()**2 + 24)/2
    p3 = ei[0]
    p4 = (5*ei.norm()**2 - 9)*ei[0]
    p5 = ei[1]
    p6 = (5*ei.norm()**2 - 9)*ei[1]
    p7 = ei[2]
    p8 = (5*ei.norm()**2 - 9)*ei[2]
    p9 = 3*ei[0]**2 - ei.norm()**2
    p10 = (3*ei.norm()**2 - 5)*(3*ei[0]**2 - ei.norm()**2)
    p11 = ei[1]**2 - ei[2]**2
    p12 = (3*ei.norm()**2 - 5)*(ei[1]**2 - ei[2]**2)
    p13 = ei[0]*ei[1]
    p14 = ei[1]*ei[2]
    p15 = ei[0]*ei[2]
    p16 = (ei[1]**2 - ei[2]**2)*ei[0]
    p17 = (ei[2]**2 - ei[0]**2)*ei[1]
    p18 = (ei[0]**2 - ei[1]**2)*ei[2]
    return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18]


M = Matrix([phi(ei.row(i)) for i in range(0, 19)]).transpose()

# def chi(ei):
#     """Transformation matrix for transition from energy PDFs to moment space
#     """
#     p0 = ei.norm()**0
#     p1 = ei[0]
#     p2 = ei[1]
#     p3 = ei[2]
#     p4 = 6 - 7*ei.norm()**2
#     p5 = 3*ei[0]**2 - ei.norm()**2
#     p6 = ei[1]**2 - ei[2]**2
#     return [p0, p1, p2, p3, p4, p5, p6]


# N = Matrix([chi(ei.row(i)) for i in range(0, 7)]).transpose()

# Transform velocity PDFs to moment space
m = M*fi

# Moment vector
code.let(rho, m[0])  # Density fluctuation
# code.let(rho, rho_0)  # Density fluctuation
code.let(en, m[1]) # Energy
code.let(epsilon, m[2]) # Energy square
code.let(jx, m[3]) # Momentum
code.let(qx, m[4]) # Energy flux
code.let(jy, m[5])
code.let(qy, m[6])
code.let(jz, m[7])
code.let(qz, m[8])
# Symmetric viscous stress tensor
code.let(pxx, m[9]/3)
code.let(pixx, m[10]/3)
code.let(pww, m[11])
code.let(piww, m[12])
code.let(pxy, m[13])
code.let(pyz, m[14])
code.let(pxz, m[15])
# Antisymmetric third-order moment
code.let(mx, m[16])
code.let(my, m[17])
code.let(mz, m[18])

mi = Matrix([rho, en, epsilon, jx, qx, jy, qy, jz, qz, 3*pxx, 3*pixx, pww, piww, pxy, pyz, pxz, mx, my, mz])

# Model stability constants
code.let(omega_e, 0)
code.let(omega_xx, 0)
code.let(omega_ej, -475/63)

# code.let(omega_e, 3)
# code.let(omega_xx, -1/2)
# code.let(omega_ej, -11/2)

code.let(vx, jx/rho)
code.let(vy, jy/rho)
code.let(vz, jz/rho)

# Moment equilibirum PDFs
m_eq0 = rho
m_eq1 = -11*rho + 19/rho_0*(jx*jx + jy*jy + jz*jz)
m_eq2 = omega_e*rho + omega_ej/rho_0*(jx*jx + jy*jy + jz*jz)
m_eq3 = jx
m_eq4 = -2/3*jx
m_eq5 = jy
m_eq6 = -2/3*jy
m_eq7 = jz
m_eq8 = -2/3*jz
m_eq9 = 1/rho_0*(2*jx*jx - (jy*jy + jz*jz))
m_eq10 = omega_xx*m_eq9
m_eq11 = (jy*jy - jz*jz)/rho_0
m_eq12 = omega_xx*m_eq11
m_eq13 = jx*jy/rho_0
m_eq14 = jy*jz/rho_0
m_eq15 = jx*jz/rho_0
m_eq16 = 0
m_eq17 = 0
m_eq18 = 0
m_eq = Matrix([m_eq0, m_eq1, m_eq2, m_eq3, m_eq4, m_eq5, m_eq6, m_eq7, m_eq8, m_eq9, m_eq10, m_eq11, m_eq12, m_eq13, m_eq14, m_eq15, m_eq16, m_eq17, m_eq18])

code.let(mi_eq, m_eq)

# Strain rate tensor
code.let(m1_1, 38/3*(jx + jy + jz))
code.let(m1_9, -2/3*(2*jx - jy - jz))
code.let(m1_11, -2/3*(jy - jz))
code.let(m1_13, -1/3*(jx + jy))
code.let(m1_14, -1/3*(jz + jy))
code.let(m1_15, -1/3*(jx + jz))
code.let(Sxx, 0 - m1_1/(38*rho_0) - m1_9/(2*rho_0))
code.let(Syy, 0 - m1_1/(38*rho_0) + m1_9/(4*rho_0) - 3*m1_11/(4*rho_0))
code.let(Szz, 0 - m1_1/(38*rho_0) + m1_9/(4*rho_0) + 3*m1_11/(4*rho_0))
code.let(Sxy, -3*m1_13/(2*rho_0))
code.let(Syz, -3*m1_14/(2*rho_0))
code.let(Sxz, -3*m1_15/(2*rho_0))
code.let(S_bar, (2*(Sxx*Sxx + Syy*Syy + Szz*Szz + Sxy*Sxy + Syz*Syz + Sxz*Sxz))**(1/2))
# Eddy viscosity
code.let(nu_t, (C*dx)**2*S_bar)

# Collision matrix for velocities in moment space
s1 = s4 = s6 = s8 = 0
s2 = 1.19
s3 = s11 = s13 = 1.4
s5 = s7 = s9 = 1.2
s17 = s18 = s19 = 1.98
s10 = s12 = s14 = s15 = s16 = 2/(1 + 6*(nu + nu_t))
S_hat = sympy.diag(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
                   s11, s12, s13, s14, s15, s16, s17, s18, s19)

# Nonequilibrium moments
code.let(mi_neq, mi - mi_eq)
code.let(mi_diff, S_hat*mi_neq)
code.let(omega, M**-1*mi_diff)

# tau = 3*nu+0.5
# S_hat = sympy.eye(19)*1/tau

# Phi = M*M.transpose()
# assert(M*(M.transpose()*Phi**-1) == sympy.eye(19,19))
# Lambda = Phi**-1*S_hat
# print(Lambda.shape)
# pprint(Lambda)
# code.let(omega, M.transpose()*(Lambda*mi_neq))

for i in range(0, 19):
    code.append(
        f'dftmp3D({i}, x, y, z, nx, ny, nz) = {fi.row(i)[0]} - {omega.row(i)[0]};')

for i in range(0, 7):
    code.append(
        f'Tdftmp3D({i}, x, y, z, nx, ny, nz) = {Ti.row(i)[0]};')

print(code)

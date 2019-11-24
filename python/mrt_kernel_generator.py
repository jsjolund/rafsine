import sympy
from sympy import Matrix, symbols
from sympy.codegen.ast import Assignment
from sympy.printing.ccode import C99CodePrinter

sympy.init_printing(use_unicode=True)

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
nu = sympy.symbols('nu')
# Thermal diffusivity
nuT = sympy.symbols('nuT')
# Smagorinsky constant
C = sympy.symbols('C')
# Turbulent Prandtl number
Pr_t = sympy.symbols('Pr_t')
# Gravity times thermal expansion
gBetta = sympy.symbols('gBetta')
# Reference temperature for Boussinesq
Tref = sympy.symbols('Tref')
# Velocity PDFs
fi = Matrix([symbols(f'f{i}') for i in range(0, 19)])
# Temperature PDFs
Ti = Matrix([symbols(f'T{i}') for i in range(0, 7)])


# Temporary variables
rho = symbols('rho')
T = symbols('T')
vx = symbols('vx')
vy = symbols('vy')
vz = symbols('vz')
V = Matrix([vx, vy, vz])
fi_eq = Matrix([symbols(f'f{i}eq') for i in range(0, 19)])
mi = Matrix([symbols(f'm{i}') for i in range(0, 19)])
mi_eq = Matrix([symbols(f'm{i}eq') for i in range(0, 19)])
mi_neq = Matrix([symbols(f'm{i}neq') for i in range(0, 19)])
omega = Matrix([symbols(f'omega{i}') for i in range(0, 19)])
code.define(rho, T, V, fi_eq, mi, mi_eq, mi_neq, omega)

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

# Compute physical quantities
code.let(rho, (sympy.ones(1, 19)*fi)[0])
code.let(T, (sympy.ones(1, 7)*Ti)[0])
code.let(vx, (Matrix([ei.row(i)[0] for i in range(0, 19)]).transpose()*fi)[0])
code.let(vy, (Matrix([ei.row(i)[1] for i in range(0, 19)]).transpose()*fi)[0])
code.let(vz, (Matrix([ei.row(i)[2] for i in range(0, 19)]).transpose()*fi)[0])


def feq(ei):
    cs2 = 1/3  # Speed of sound squared
    return rho*e_omega*(1 + ei.dot(V)/cs2 + (ei.dot(V))**2/(2*cs2**2) - V.dot(V)/(2*cs2))


# Equilibrium PDFs in velocity space
code.let(fi_eq, sympy.Matrix([feq(ei.row(i)) for i in range(0, 19)]))


def phi(ei):
    """Transformation matrix for transition from velocity to moment space
    """
    p0 = ei.norm()**0
    p1 = 19*ei.norm()**2 - 30
    p2 = (21*ei.norm()**4 - 53*ei.norm()**2 + 24)/2
    p3 = ei[0]
    p5 = ei[1]
    p7 = ei[2]
    p4 = (5*ei.norm()**2 - 9)*ei[0]
    p6 = (5*ei.norm()**2 - 9)*ei[1]
    p8 = (5*ei.norm()**2 - 9)*ei[2]
    p9 = 3*ei[0]**2 - ei.norm()**2
    p11 = ei[1]**2 - ei[2]**2
    p13 = ei[0]*ei[1]
    p14 = ei[1]*ei[2]
    p15 = ei[0]*ei[2]
    p10 = (3*ei.norm()**2 - 5)*(3*ei[0]**2 - ei.norm()**2)
    p12 = (3*ei.norm()**2 - 5)*(ei[1]**2 - ei[2]**2)
    p16 = (ei[1]**2 - ei[2]**2)*ei[0]
    p17 = (ei[2]**2 - ei[0]**2)*ei[1]
    p18 = (ei[0]**2 - ei[1]**2)*ei[2]
    return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18]


M = Matrix([phi(ei.row(i)) for i in range(0, 19)]).transpose()
M_inv = M**-1

# Collision matrix for velocities in moment space
s1 = s4 = s6 = s8 = 0
s2 = 1.19
s3 = s11 = s13 = 1.4
s5 = s7 = s9 = 1.2
s17 = s18 = s19 = 1.98
s10 = s12 = s14 = s15 = s16 = 2/(1 + 6*nu)
S_hat = sympy.diag(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
                   s11, s12, s13, s14, s15, s16, s17, s18, s19)


def chi(ei):
    """Transformation matrix for transition from energy PDFs to moment space
    """
    p0 = ei.norm()**0
    p1 = ei[0]
    p2 = ei[1]
    p3 = ei[2]
    p4 = 6 - 7*ei.norm()**2
    p5 = 3*ei[0]**2 - ei.norm()**2
    p6 = ei[1]**2 - ei[2]**2
    return [p0, p1, p2, p3, p4, p5, p6]


N = Matrix([chi(ei.row(i)) for i in range(0, 7)]).transpose()
N_inv = N**-1


def eq(m):
    """Calculate equlilibrium PDFs in moment space
    """
    rho = m[0]  # Density
    en = m[1]  # Energy
    epsilon = m[2]  # Energy square
    jx = m[3]  # Momentum
    jy = m[5]
    jz = m[7]
    qx = m[4]  # Energy flux
    qy = m[6]
    qz = m[8]
    # Symmetric viscous stress tensor
    pxx = m[9]/3
    pixx = m[10]/3
    pww = m[11]
    piww = m[12]
    pxy = m[13]
    pyz = m[14]
    pxz = m[15]
    # Antisymmetric third-order moment
    mx = m[16]
    my = m[17]
    mz = m[18]
    # Model stability constants
    omega_e = 0
    omega_xx = 0
    omega_ej = -475/63
    rho_0 = 1  # Mean density in system
    # Output
    rho_eq = rho
    en_eq = -11*rho + 19/rho_0*(jx*jx + jy*jy + jz*jz)
    epsilon_eq = omega_e*rho + omega_ej/rho_0*(jx*jx + jy*jy + jz*jz)
    jx_eq = jx
    qx_eq = -2/3*jx
    jy_eq = jy
    qy_eq = -2/3*jy
    jz_eq = jz
    qz_eq = -2/3*jz
    pxx_eq = 1/rho_0*(2*jx*jx - (jy*jy + jz*jz))
    pixx_eq = omega_xx*1/rho_0*(2*jx*jx - (jy*jy + jz*jz))
    pww_eq = 1/rho_0*(jy*jy - jz*jz)
    piww_eq = omega_xx/rho_0*(jy*jy - jz*jz)
    pxy_eq = jx*jy/rho_0
    pyz_eq = jy*jz/rho_0
    pxz_eq = jx*jz/rho_0
    mx_eq = 0
    my_eq = 0
    mz_eq = 0
    return Matrix([rho_eq, en_eq, epsilon_eq, jx_eq, qx_eq, jy_eq, qy_eq, jz_eq, qz_eq, pxx_eq, pixx_eq, pww_eq, piww_eq, pxy_eq, pyz_eq, pxz_eq, mx_eq, my_eq, mz_eq])


# Transform velocity PDFs to moment space
m = M*fi
code.let(mi, m)
# Velocity moments equilibirum PDFs
m_eq = eq(m)
code.let(mi_eq, m_eq)
# Nonequilibrium moments
code.let(mi_neq, m - m_eq)

code.let(omega, -M_inv*(S_hat*mi_neq))

for i in range(0, 19):
    code.append(
        f'dftmp3D({i}, x, y, z, nx, ny, nz) = {fi.row(i)[0]} + {omega.row(i)[0]};')

for i in range(0, 7):
    code.append(
        f'Tdftmp3D({i}, x, y, z, nx, ny, nz) = {Ti.row(i)[0]};')

print(code)

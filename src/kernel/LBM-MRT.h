real m0eq, m1eq, m2eq, m3eq, m4eq, m5eq, m6eq, m7eq, m8eq, m9eq, m10eq, m11eq, m12eq, m13eq, m14eq, m15eq, m16eq, m17eq, m18eq;
real m0neq, m1neq, m2neq, m3neq, m4neq, m5neq, m6neq, m7neq, m8neq, m9neq, m10neq, m11neq, m12neq, m13neq, m14neq, m15neq, m16neq, m17neq, m18neq;
real omega0, omega1, omega2, omega3, omega4, omega5, omega6, omega7, omega8, omega9, omega10, omega11, omega12, omega13, omega14, omega15, omega16, omega17, omega18;
real n0eq, n1eq, n2eq, n3eq, n4eq, n5eq, n6eq;
real n0neq, n1neq, n2neq, n3neq, n4neq, n5neq, n6neq;
real omegaT0, omegaT1, omegaT2, omegaT3, omegaT4, omegaT5, omegaT6;
real S_bar;
real ST;
real T;
real Fup;
real Fdown;
real tau_V;
real tau_T;
real Sxx, Syy, Szz, Sxy, Syz, Sxz;
real m1_1, m1_9, m1_11, m1_13, m1_14, m1_15;
real rho, en, epsilon, jx, qx, jy, qy, jz, qz, pxx3, pixx3, pww, piww, pxy, pyz, pxz, mx, my, mz;
real omega_e, omega_xx, omega_ej;
real vx, vy, vz;

// Macroscopic density
rho = f0 + f1 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9;

// Part of kinetic energy independent of density
en = -30.0*f0 - 11.0*f1 + 8.0*f10 + 8.0*f11 + 8.0*f12 + 8.0*f13 + 8.0*f14 + 8.0*f15 + 8.0*f16 + 8.0*f17 + 8.0*f18 - 11.0*f2 - 11.0*f3 - 11.0*f4 - 11.0*f5 - 11.0*f6 + 8.0*f7 + 8.0*f8 + 8.0*f9;

// Kinetic energy square
epsilon = 12.0*f0 - 4.0*f1 + 1.0*f10 + 1.0*f11 + 1.0*f12 + 1.0*f13 + 1.0*f14 + 1.0*f15 + 1.0*f16 + 1.0*f17 + 1.0*f18 - 4.0*f2 - 4.0*f3 - 4.0*f4 - 4.0*f5 - 4.0*f6 + 1.0*f7 + 1.0*f8 + 1.0*f9;

// Momentum
jx = f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9;
jy = f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9;
jz = f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6;

// Energy flux independent of mass flux
qx = -4.0*f1 - 1.0*f10 + 1.0*f11 - 1.0*f12 + 1.0*f13 - 1.0*f14 + 4.0*f2 + 1.0*f7 - 1.0*f8 + 1.0*f9;
qy = 1.0*f10 + 1.0*f15 - 1.0*f16 + 1.0*f17 - 1.0*f18 - 4.0*f3 + 4.0*f4 + 1.0*f7 - 1.0*f8 - 1.0*f9;
qz = 1.0*f11 - 1.0*f12 - 1.0*f13 + 1.0*f14 + 1.0*f15 - 1.0*f16 - 1.0*f17 + 1.0*f18 - 4.0*f5 + 4.0*f6;

// Symmetric traceless viscous stress tensor
pxx3 = 2.0*f1 + 1.0*f10 + 1.0*f11 + 1.0*f12 + 1.0*f13 + 1.0*f14 - 2*f15 - 2*f16 - 2*f17 - 2*f18 + 2.0*f2 - f3 - f4 - f5 - f6 + 1.0*f7 + 1.0*f8 + 1.0*f9;
pww = f10 - f11 - f12 - f13 - f14 + f3 + f4 - f5 - f6 + f7 + f8 + f9;
pxy = -f10 + f7 + f8 - f9;
pyz = f15 + f16 - f17 - f18;
pxz = f11 + f12 - f13 - f14;

// Fourth order moments
pixx3 = -4.0*f1 + 1.0*f10 + 1.0*f11 + 1.0*f12 + 1.0*f13 + 1.0*f14 - 2.0*f15 - 2.0*f16 - 2.0*f17 - 2.0*f18 - 4.0*f2 + 2.0*f3 + 2.0*f4 + 2.0*f5 + 2.0*f6 + 1.0*f7 + 1.0*f8 + 1.0*f9;
piww = 1.0*f10 - 1.0*f11 - 1.0*f12 - 1.0*f13 - 1.0*f14 - 2.0*f3 - 2.0*f4 + 2.0*f5 + 2.0*f6 + 1.0*f7 + 1.0*f8 + 1.0*f9;

// Antisymmetric third-order moment
mx = -f10 - f11 + f12 - f13 + f14 + f7 - f8 + f9;
my = -f10 + f15 - f16 + f17 - f18 - f7 + f8 + f9;
mz = f11 - f12 - f13 + f14 - f15 + f16 + f17 - f18;

// Model stability constants
omega_e = 0;
omega_xx = 0;
omega_ej = -7.5396825396825395;

// Macroscopic velocity
vx = jx/rho;
vy = jy/rho;
vz = jz/rho;

// Velocity moment equilibirum distribution functions
m0eq = rho;
m1eq = 19.0*pow(jx, 2) + 19.0*pow(jy, 2) + 19.0*pow(jz, 2) - 11.0*rho;
m2eq = omega_e*rho + 1.0*omega_ej*(pow(jx, 2) + pow(jy, 2) + pow(jz, 2));
m3eq = jx;
m4eq = -0.66666666666666663*jx;
m5eq = jy;
m6eq = -0.66666666666666663*jy;
m7eq = jz;
m8eq = -0.66666666666666663*jz;
m9eq = 2.0*pow(jx, 2) - 1.0*pow(jy, 2) - 1.0*pow(jz, 2);
m10eq = omega_xx*(2.0*pow(jx, 2) - 1.0*pow(jy, 2) - 1.0*pow(jz, 2));
m11eq = 1.0*pow(jy, 2) - 1.0*pow(jz, 2);
m12eq = omega_xx*(1.0*pow(jy, 2) - 1.0*pow(jz, 2));
m13eq = 1.0*jx*jy;
m14eq = 1.0*jy*jz;
m15eq = 1.0*jx*jz;
m16eq = 0;
m17eq = 0;
m18eq = 0;

// LES strain rate tensor
m1_1 = 12.666666666666666*jx + 12.666666666666666*jy + 12.666666666666666*jz;
m1_9 = -1.3333333333333333*jx + 0.66666666666666663*jy + 0.66666666666666663*jz;
m1_11 = -0.66666666666666663*jy + 0.66666666666666663*jz;
m1_13 = -0.33333333333333331*jx - 0.33333333333333331*jy;
m1_14 = -0.33333333333333331*jy - 0.33333333333333331*jz;
m1_15 = -0.33333333333333331*jx - 0.33333333333333331*jz;
Sxx = -0.026315789473684209*m1_1 - 0.5*m1_9;
Syy = -0.026315789473684209*m1_1 - 0.75*m1_11 + 0.25*m1_9;
Szz = -0.026315789473684209*m1_1 + 0.75*m1_11 + 0.25*m1_9;
Sxy = -1.5*m1_13;
Syz = -1.5*m1_14;
Sxz = -1.5*m1_15;

// Magnitude of strain rate tensor
S_bar = 1.4142135623730951*sqrt(pow(Sxx, 2) + pow(Sxy, 2) + pow(Sxz, 2) + pow(Syy, 2) + pow(Syz, 2) + pow(Szz, 2));

// Filtered strain rate
ST = 4.0*pow(C, 2.0)*S_bar;

// Macroscopic temperature
T = T0 + T1 + T2 + T3 + T4 + T5 + T6;

// Temperature moment equilibirum distribution functions
n0eq = T;
n1eq = T*vx;
n2eq = T*vy;
n3eq = T*vz;
n4eq = 0.75*T;
n5eq = 0.0;
n6eq = 0.0;

// Boussinesq approximation of body force
Fup = 0.055555555555555552*gBetta*rho*(1 - vz)*(T - Tref)*(-1.5*pow(vx, 2) - 1.5*pow(vy, 2) + 3.0*pow(vz, 2) + 3.0*vz + 1);
Fdown = 0.055555555555555552*gBetta*rho*(T - Tref)*(-vz - 1)*(-1.5*pow(vx, 2) - 1.5*pow(vy, 2) + 3.0*pow(vz, 2) - 3.0*vz + 1);

// Modified heat diffusion
tau_T = 9.5/(20.0*nuT + 4.75 + 20.0*ST/Pr_t);

// Modified shear viscosity
tau_V = 1.0/(3.0*ST + 3.0*nu + 0.5);

// Difference to velocity equilibrium
m0neq = -m0eq + rho;
m1neq = en - m1eq;
m2neq = epsilon - m2eq;
m3neq = jx - m3eq;
m4neq = -m4eq + qx;
m5neq = jy - m5eq;
m6neq = -m6eq + qy;
m7neq = jz - m7eq;
m8neq = -m8eq + qz;
m9neq = -m9eq + pxx3;
m10neq = -m10eq + pixx3;
m11neq = -m11eq + pww;
m12neq = -m12eq + piww;
m13neq = -m13eq + pxy;
m14neq = -m14eq + pyz;
m15neq = -m15eq + pxz;
m16neq = -m16eq + mx;
m17neq = -m17eq + my;
m18neq = -m18eq + mz;

// Relax velocity
omega0 = -0.014912280701754385*m1neq + 0.066666666666666652*m2neq;
omega1 = -0.077777777777777765*m10neq - 0.0054678362573099409*m1neq - 0.02222222222222222*m2neq - 0.12*m4neq + 0.055555555555555552*m9neq*tau_V;
omega2 = -0.077777777777777765*m10neq - 0.0054678362573099409*m1neq - 0.02222222222222222*m2neq + 0.12*m4neq + 0.055555555555555552*m9neq*tau_V;
omega3 = 0.038888888888888883*m10neq + (1.0/12.0)*m11neq*tau_V - 0.11666666666666665*m12neq - 0.0054678362573099409*m1neq - 0.02222222222222222*m2neq - 0.12*m6neq - 0.027777777777777776*m9neq*tau_V;
omega4 = 0.038888888888888883*m10neq + (1.0/12.0)*m11neq*tau_V - 0.11666666666666665*m12neq - 0.0054678362573099409*m1neq - 0.02222222222222222*m2neq + 0.12*m6neq - 0.027777777777777776*m9neq*tau_V;
omega5 = 0.038888888888888883*m10neq - 1.0/12.0*m11neq*tau_V + 0.11666666666666665*m12neq - 0.0054678362573099409*m1neq - 0.02222222222222222*m2neq - 0.12*m8neq - 0.027777777777777776*m9neq*tau_V;
omega6 = 0.038888888888888883*m10neq - 1.0/12.0*m11neq*tau_V + 0.11666666666666665*m12neq - 0.0054678362573099409*m1neq - 0.02222222222222222*m2neq + 0.12*m8neq - 0.027777777777777776*m9neq*tau_V;
omega7 = 0.019444444444444441*m10neq + (1.0/12.0)*m11neq*tau_V + 0.058333333333333327*m12neq + (1.0/4.0)*m13neq*tau_V + 0.2475*m16neq - 0.2475*m17neq + 0.0039766081871345027*m1neq + 0.0055555555555555549*m2neq + 0.029999999999999999*m4neq + 0.029999999999999999*m6neq + 0.027777777777777776*m9neq*tau_V;
omega8 = 0.019444444444444441*m10neq + (1.0/12.0)*m11neq*tau_V + 0.058333333333333327*m12neq + (1.0/4.0)*m13neq*tau_V - 0.2475*m16neq + 0.2475*m17neq + 0.0039766081871345027*m1neq + 0.0055555555555555549*m2neq - 0.029999999999999999*m4neq - 0.029999999999999999*m6neq + 0.027777777777777776*m9neq*tau_V;
omega9 = 0.019444444444444441*m10neq + (1.0/12.0)*m11neq*tau_V + 0.058333333333333327*m12neq - 1.0/4.0*m13neq*tau_V + 0.2475*m16neq + 0.2475*m17neq + 0.0039766081871345027*m1neq + 0.0055555555555555549*m2neq + 0.029999999999999999*m4neq - 0.029999999999999999*m6neq + 0.027777777777777776*m9neq*tau_V;
omega10 = 0.019444444444444441*m10neq + (1.0/12.0)*m11neq*tau_V + 0.058333333333333327*m12neq - 1.0/4.0*m13neq*tau_V - 0.2475*m16neq - 0.2475*m17neq + 0.0039766081871345027*m1neq + 0.0055555555555555549*m2neq - 0.029999999999999999*m4neq + 0.029999999999999999*m6neq + 0.027777777777777776*m9neq*tau_V;
omega11 = 0.019444444444444441*m10neq - 1.0/12.0*m11neq*tau_V - 0.058333333333333327*m12neq + (1.0/4.0)*m15neq*tau_V - 0.2475*m16neq + 0.2475*m18neq + 0.0039766081871345027*m1neq + 0.0055555555555555549*m2neq + 0.029999999999999999*m4neq + 0.029999999999999999*m8neq + 0.027777777777777776*m9neq*tau_V;
omega12 = 0.019444444444444441*m10neq - 1.0/12.0*m11neq*tau_V - 0.058333333333333327*m12neq + (1.0/4.0)*m15neq*tau_V + 0.2475*m16neq - 0.2475*m18neq + 0.0039766081871345027*m1neq + 0.0055555555555555549*m2neq - 0.029999999999999999*m4neq - 0.029999999999999999*m8neq + 0.027777777777777776*m9neq*tau_V;
omega13 = 0.019444444444444441*m10neq - 1.0/12.0*m11neq*tau_V - 0.058333333333333327*m12neq - 1.0/4.0*m15neq*tau_V - 0.2475*m16neq - 0.2475*m18neq + 0.0039766081871345027*m1neq + 0.0055555555555555549*m2neq + 0.029999999999999999*m4neq - 0.029999999999999999*m8neq + 0.027777777777777776*m9neq*tau_V;
omega14 = 0.019444444444444441*m10neq - 1.0/12.0*m11neq*tau_V - 0.058333333333333327*m12neq - 1.0/4.0*m15neq*tau_V + 0.2475*m16neq + 0.2475*m18neq + 0.0039766081871345027*m1neq + 0.0055555555555555549*m2neq - 0.029999999999999999*m4neq + 0.029999999999999999*m8neq + 0.027777777777777776*m9neq*tau_V;
omega15 = -0.038888888888888883*m10neq + (1.0/4.0)*m14neq*tau_V + 0.2475*m17neq - 0.2475*m18neq + 0.0039766081871345027*m1neq + 0.0055555555555555549*m2neq + 0.029999999999999999*m6neq + 0.029999999999999999*m8neq - 0.055555555555555552*m9neq*tau_V;
omega16 = -0.038888888888888883*m10neq + (1.0/4.0)*m14neq*tau_V - 0.2475*m17neq + 0.2475*m18neq + 0.0039766081871345027*m1neq + 0.0055555555555555549*m2neq - 0.029999999999999999*m6neq - 0.029999999999999999*m8neq - 0.055555555555555552*m9neq*tau_V;
omega17 = -0.038888888888888883*m10neq - 1.0/4.0*m14neq*tau_V + 0.2475*m17neq + 0.2475*m18neq + 0.0039766081871345027*m1neq + 0.0055555555555555549*m2neq + 0.029999999999999999*m6neq - 0.029999999999999999*m8neq - 0.055555555555555552*m9neq*tau_V;
omega18 = -0.038888888888888883*m10neq - 1.0/4.0*m14neq*tau_V - 0.2475*m17neq - 0.2475*m18neq + 0.0039766081871345027*m1neq + 0.0055555555555555549*m2neq - 0.029999999999999999*m6neq + 0.029999999999999999*m8neq - 0.055555555555555552*m9neq*tau_V;

// Difference to temperature equilibrium
n0neq = T0 + T1 + T2 + T3 + T4 + T5 + T6 - n0eq;
n1neq = T1 - T2 - n1eq;
n2neq = T3 - T4 - n2eq;
n3neq = T5 - T6 - n3eq;
n4neq = 6.0*T0 - 1.0*T1 - 1.0*T2 - 1.0*T3 - 1.0*T4 - 1.0*T5 - 1.0*T6 - n4eq;
n5neq = 2.0*T1 + 2.0*T2 - T3 - T4 - T5 - T6 - n5eq;
n6neq = T3 + T4 - T5 - T6 - n6eq;

// Relax temperature
omegaT0 = 0.14285714285714285*n4neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T));
omegaT1 = 0.5*n1neq*tau_T - 0.023809523809523808*n4neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T)) + 0.16666666666666666*n5neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T));
omegaT2 = -0.5*n1neq*tau_T - 0.023809523809523808*n4neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T)) + 0.16666666666666666*n5neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T));
omegaT3 = 0.5*n2neq*tau_T - 0.023809523809523808*n4neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T)) - 0.083333333333333329*n5neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T)) + 0.25*n6neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T));
omegaT4 = -0.5*n2neq*tau_T - 0.023809523809523808*n4neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T)) - 0.083333333333333329*n5neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T)) + 0.25*n6neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T));
omegaT5 = 0.5*n3neq*tau_T - 0.023809523809523808*n4neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T)) - 0.083333333333333329*n5neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T)) - 0.25*n6neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T));
omegaT6 = -0.5*n3neq*tau_T - 0.023809523809523808*n4neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T)) - 0.083333333333333329*n5neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T)) - 0.25*n6neq/(0.5 + 1.0/(-3.0 + 6.0/tau_T));

// Write relaxed velocity
dftmp3D(0, x, y, z, nx, ny, nz) = f0 - omega0;
dftmp3D(1, x, y, z, nx, ny, nz) = f1 - omega1;
dftmp3D(2, x, y, z, nx, ny, nz) = f2 - omega2;
dftmp3D(3, x, y, z, nx, ny, nz) = f3 - omega3;
dftmp3D(4, x, y, z, nx, ny, nz) = f4 - omega4;
dftmp3D(5, x, y, z, nx, ny, nz) = Fup + f5 - omega5;
dftmp3D(6, x, y, z, nx, ny, nz) = Fdown + f6 - omega6;
dftmp3D(7, x, y, z, nx, ny, nz) = f7 - omega7;
dftmp3D(8, x, y, z, nx, ny, nz) = f8 - omega8;
dftmp3D(9, x, y, z, nx, ny, nz) = f9 - omega9;
dftmp3D(10, x, y, z, nx, ny, nz) = f10 - omega10;
dftmp3D(11, x, y, z, nx, ny, nz) = f11 - omega11;
dftmp3D(12, x, y, z, nx, ny, nz) = f12 - omega12;
dftmp3D(13, x, y, z, nx, ny, nz) = f13 - omega13;
dftmp3D(14, x, y, z, nx, ny, nz) = f14 - omega14;
dftmp3D(15, x, y, z, nx, ny, nz) = f15 - omega15;
dftmp3D(16, x, y, z, nx, ny, nz) = f16 - omega16;
dftmp3D(17, x, y, z, nx, ny, nz) = f17 - omega17;
dftmp3D(18, x, y, z, nx, ny, nz) = f18 - omega18;

// Write relaxed temperature
Tdftmp3D(0, x, y, z, nx, ny, nz) = T0 - omegaT0;
Tdftmp3D(1, x, y, z, nx, ny, nz) = T1 - omegaT1;
Tdftmp3D(2, x, y, z, nx, ny, nz) = T2 - omegaT2;
Tdftmp3D(3, x, y, z, nx, ny, nz) = T3 - omegaT3;
Tdftmp3D(4, x, y, z, nx, ny, nz) = T4 - omegaT4;
Tdftmp3D(5, x, y, z, nx, ny, nz) = T5 - omegaT5;
Tdftmp3D(6, x, y, z, nx, ny, nz) = T6 - omegaT6;
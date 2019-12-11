// Compute physical quantities
const real rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 +
                 f12 + f13 + f14 + f15 + f16 + f17 + f18;
const real T = T0 + T1 + T2 + T3 + T4 + T5 + T6;
const real vx =
    (1 / rho) * (f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14);
const real vy =
    (1 / rho) * (f3 - f4 + f7 - f8 - f9 + f10 + f15 - f16 + f17 - f18);
const real vz =
    (1 / rho) * (f5 - f6 + f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18);

// Compute the equilibrium distribution function
const real sq_term = -1.5f * (vx * vx + vy * vy + vz * vz);
const real f0eq = rho * (1.f / 3.f) * (1 + sq_term);
const real f1eq = rho * (1.f / 18.f) * (1 + 3 * vx + 4.5f * vx * vx + sq_term);
const real f2eq = rho * (1.f / 18.f) * (1 - 3 * vx + 4.5f * vx * vx + sq_term);
const real f3eq = rho * (1.f / 18.f) * (1 + 3 * vy + 4.5f * vy * vy + sq_term);
const real f4eq = rho * (1.f / 18.f) * (1 - 3 * vy + 4.5f * vy * vy + sq_term);
const real f5eq = rho * (1.f / 18.f) * (1 + 3 * vz + 4.5f * vz * vz + sq_term);
const real f6eq = rho * (1.f / 18.f) * (1 - 3 * vz + 4.5f * vz * vz + sq_term);
const real f7eq = rho * (1.f / 36.f) *
                  (1 + 3 * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
const real f8eq = rho * (1.f / 36.f) *
                  (1 - 3 * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
const real f9eq = rho * (1.f / 36.f) *
                  (1 + 3 * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
const real f10eq = rho * (1.f / 36.f) *
                   (1 - 3 * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
const real f11eq = rho * (1.f / 36.f) *
                   (1 + 3 * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
const real f12eq = rho * (1.f / 36.f) *
                   (1 - 3 * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
const real f13eq = rho * (1.f / 36.f) *
                   (1 + 3 * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
const real f14eq = rho * (1.f / 36.f) *
                   (1 - 3 * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
const real f15eq = rho * (1.f / 36.f) *
                   (1 + 3 * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
const real f16eq = rho * (1.f / 36.f) *
                   (1 - 3 * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
const real f17eq = rho * (1.f / 36.f) *
                   (1 + 3 * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);
const real f18eq = rho * (1.f / 36.f) *
                   (1 - 3 * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);

// Compute the equilibrium temperature distribution
const real T0eq = T * (1.f / 7.f);
const real T1eq = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vx);
const real T2eq = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vx);
const real T3eq = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vy);
const real T4eq = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vy);
const real T5eq = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vz);
const real T6eq = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vz);

// Difference to equilibrium
const real f1diff = f1 - f1eq;
const real f2diff = f2 - f2eq;
const real f3diff = f3 - f3eq;
const real f4diff = f4 - f4eq;
const real f5diff = f5 - f5eq;
const real f6diff = f6 - f6eq;
const real f7diff = f7 - f7eq;
const real f8diff = f8 - f8eq;
const real f9diff = f9 - f9eq;
const real f10diff = f10 - f10eq;
const real f11diff = f11 - f11eq;
const real f12diff = f12 - f12eq;
const real f13diff = f13 - f13eq;
const real f14diff = f14 - f14eq;
const real f15diff = f15 - f15eq;
const real f16diff = f16 - f16eq;
const real f17diff = f17 - f17eq;
const real f18diff = f18 - f18eq;

// Non equilibrium stress-tensor for velocity
const real Pi_x_x = f1diff + f2diff + f7diff + f8diff + f9diff + f10diff +
                    f11diff + f12diff + f13diff + f14diff;
const real Pi_x_y = f7diff + f8diff - f9diff - f10diff;
const real Pi_x_z = f11diff + f12diff - f13diff - f14diff;
const real Pi_y_y = f3diff + f4diff + f7diff + f8diff + f9diff + f10diff +
                    f15diff + f16diff + f17diff + f18diff;
const real Pi_y_z = f15diff + f16diff - f17diff - f18diff;
const real Pi_z_z = f5diff + f6diff + f11diff + f12diff + f13diff + f14diff +
                    f15diff + f16diff + f17diff + f18diff;
// Variance
const real Q = Pi_x_x * Pi_x_x + 2 * Pi_x_y * Pi_x_y + 2 * Pi_x_z * Pi_x_z +
               Pi_y_y * Pi_y_y + 2 * Pi_y_z * Pi_y_z + Pi_z_z * Pi_z_z;
// Local stress tensor
const real ST = (1 / (real)6) * (sqrt(nu * nu + 18 * C * C * sqrt(Q)) - nu);
// Modified relaxation time
const real tau = 3 * (nu + ST) + (real)0.5;

dftmp3D(0, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f0 + (1 / tau) * f0eq;
dftmp3D(1, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f1 + (1 / tau) * f1eq;
dftmp3D(2, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f2 + (1 / tau) * f2eq;
dftmp3D(3, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f3 + (1 / tau) * f3eq;
dftmp3D(4, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f4 + (1 / tau) * f4eq;
dftmp3D(5, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f5 + (1 / tau) * f5eq +
                                  0.5f * gBetta * (T - Tref);
dftmp3D(6, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f6 + (1 / tau) * f6eq -
                                  0.5f * gBetta * (T - Tref);
dftmp3D(7, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f7 + (1 / tau) * f7eq;
dftmp3D(8, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f8 + (1 / tau) * f8eq;
dftmp3D(9, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f9 + (1 / tau) * f9eq;
dftmp3D(10, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f10 + (1 / tau) * f10eq;
dftmp3D(11, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f11 + (1 / tau) * f11eq;
dftmp3D(12, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f12 + (1 / tau) * f12eq;
dftmp3D(13, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f13 + (1 / tau) * f13eq;
dftmp3D(14, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f14 + (1 / tau) * f14eq;
dftmp3D(15, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f15 + (1 / tau) * f15eq;
dftmp3D(16, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f16 + (1 / tau) * f16eq;
dftmp3D(17, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f17 + (1 / tau) * f17eq;
dftmp3D(18, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f18 + (1 / tau) * f18eq;

// Modified relaxation time for the temperature
const real tauT = 3 * (nuT + ST / Pr_t) + (real)0.5;

// Relax temperature
Tdftmp3D(0, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T0 + (1 / tauT) * T0eq;
Tdftmp3D(1, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T1 + (1 / tauT) * T1eq;
Tdftmp3D(2, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T2 + (1 / tauT) * T2eq;
Tdftmp3D(3, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T3 + (1 / tauT) * T3eq;
Tdftmp3D(4, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T4 + (1 / tauT) * T4eq;
Tdftmp3D(5, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T5 + (1 / tauT) * T5eq;
Tdftmp3D(6, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T6 + (1 / tauT) * T6eq;

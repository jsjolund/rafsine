package.path = package.path .. ";lua/?.lua"
require "unitConverter"

-- Physical Settings
uc =
  UnitConverter(
    {
      reference_length_in_meters = 6.95,
      reference_length_in_number_of_nodes = 256,
      reference_speed_in_meters_per_second = 1.0,
      reference_speed_in_lattice_units = 0.1,
      temperature_conversion_factor = 1,
      reference_temperature_physical = 0,
      reference_temperature_LBM = 0
    }
  )

-- Velocity conversion factor
C_U = uc.C_U

-- Reference length
L_phys = uc.ref_L_phys

-- Size in meters
mx = 6.95
my = 6.4
mz = 3.1

-- Size of the lattice
nx = uc:m_to_lu(mx) + 1
ny = uc:m_to_lu(my) + 1
nz = uc:m_to_lu(mz) + 1

-- Kinematic viscosity of air
nu = uc:Nu_to_lu(1.568e-5)
-- Thermal diffusivity
nuT = 1.0e-2
-- nuT = uc:Nu_to_lu(21.70e-6)
-- Smagorinsky constant
C = 0.02
-- Thermal conductivity
k = 2.624e-5
-- Prandtl number of air
Pr = 0.707
-- Turbulent Prandtl number
Pr_t = 0.9

-- Gravity * thermal expansion
gBetta = uc:gBetta_to_lu(9.82 * 3.32e-3)

-- Initial temperature
Tinit = uc:Temp_to_lu(18)
-- Reference temperature
Tref = Tinit

-- Averaging period in seconds
avgPeriod = 30.0

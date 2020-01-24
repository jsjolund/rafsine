package.path = package.path .. "./?.lua;lua/?.lua"
require "UnitConverter"

-- Physical Settings --
uc = UnitConverter(
  {
    reference_length_in_meters = 7.2,
    reference_length_in_number_of_nodes = 128,
    reference_speed_in_meters_per_second = 1.0,
    reference_speed_in_lattice_units = 0.1,
    temperature_conversion_factor = 1,
    reference_temperature_physical = 0,
    reference_temperature_LBM = 0
  }
)

-- Velocity conversion factor
C_U = uc.C_U
C_L = uc.C_L

-- Size of the lattice
nx = uc:m_to_lu(7.2) + 1
ny = uc:m_to_lu(7.2) + 1
nz = uc:m_to_lu(3.0) + 1

-- Viscosity
nu = uc:Nu_to_lu(1.511e-5)

-- Smagorinsky constant
C = 0.18
-- Thermal diffusivity
nuT = 1.0e-2

-- Turbulent Prandtl number
Pr_t = 0.9

-- Gravity * thermal expansion
--gBetta = uc:gBetta_to_lu(3.0e-3)
gBetta = 1e-5

-- Initial temperature
Tinit = uc:Temp_to_lu(16)
-- Reference temperature
Tref = Tinit

-- Averaging period in seconds
avgPeriod = 10.0

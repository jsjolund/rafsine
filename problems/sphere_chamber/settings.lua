package.path = package.path .. ";lua/?.lua"
require "UnitConverter"

-- Physical Settings --
uc = UnitConverter(
  {
    reference_length_in_meters = 5.0,
    -- reference_length_in_number_of_nodes = 32,
    reference_length_in_number_of_nodes = 96,
    -- reference_length_in_number_of_nodes = 128,
    -- reference_length_in_number_of_nodes = 192,
    -- reference_length_in_number_of_nodes = 256,
    -- reference_length_in_number_of_nodes = 512,
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

-- Size in meters
mx = 5.0
my = 20.0
mz = 5.0

-- Size of the lattice
nx = uc:m_to_lu(mx) + 1
ny = uc:m_to_lu(my) + 1
nz = uc:m_to_lu(mz) + 1

-- Smagorinsky constant
-- C = 0.02 --BGK
C = 0.1

-- Kinematic viscosity of air
nu = uc:Nu_to_lu(1.506e-5)
-- Thermal diffusivity
nuT = uc:Nu_to_lu(21.70e-6)

-- Turbulent Prandtl number
Pr_t = 0.9

-- Gravity * thermal expansion
--gBetta = uc:gBetta_to_lu(3.0e-3)
gBetta = 0

-- Initial temperature
Tinit = uc:Temp_to_lu(10)
-- Reference temperature
Tref = Tinit

-- Averaging period in seconds
avgPeriod = 10

-- Partitioning axis for multiple GPUs
partitioning = 'Y'

-- LBM method
method = 'BGK'
-- method = 'MRT'

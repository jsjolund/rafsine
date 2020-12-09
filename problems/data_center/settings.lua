package.path = package.path .. ";lua/?.lua"
require "UnitConverter"

-- Physical Settings --
uc =
  UnitConverter(
    {
      reference_length_in_meters = 4.8,
      -- reference_length_in_number_of_nodes = 720,
      -- reference_length_in_number_of_nodes = 512,
      -- reference_length_in_number_of_nodes = 256,
      reference_length_in_number_of_nodes = 128,
      -- reference_length_in_number_of_nodes = 64,
      reference_speed_in_meters_per_second = 1.0,
      reference_speed_in_lattice_units = 0.1,
      temperature_conversion_factor = 1,
      reference_temperature_physical = 0,
      reference_temperature_LBM = 0
    }
  )

-- Velocity conversion factor
C_U = uc.C_U

-- Size of the lattice
nx = uc:m_to_lu(4.8) + 1
ny = uc:m_to_lu(6.0) + 1
nz = uc:m_to_lu(2.8) + 1

-- Viscosity
nu = uc:Nu_to_lu(1.511e-5)

-- Smagorinsky constant
C = 0.04
-- C = 0.18

-- Thermal diffusivity
nuT = 1.0e-2

-- Turbulent Prandtl number
Pr_t = 0.9

-- Gravity * thermal expansion
gBetta = uc:gBetta_to_lu(3.0e-6)

-- Initial temperature
Tinit = uc:Temp_to_lu(16 + 5.6216 / 2)
-- Reference temperature
Tref = Tinit

-- Averaging period in seconds
avgPeriod = 10.0

-- Partitioning axis for multiple GPUs
partitioning = 'Y'

-- LBM method
method = 'BGK'
-- method = 'MRT'

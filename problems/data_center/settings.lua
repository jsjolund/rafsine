package.path = package.path .. ";lua/?.lua"
require "unitConverter"

-- Physical Settings --

uc =
    UnitConverter(
    {
        reference_length_in_meters = 4.8,
        reference_length_in_number_of_nodes = 128,
        reference_speed_in_meters_per_second = 1.0,
        reference_speed_in_lattice_units = 0.1,
        temperature_convertion_factor = 1,
        reference_temperature_physical = 0,
        reference_temperature_LBM = 0
    }
)

-- velocity conversion factor
C_U = uc.C_U

-- size of the lattice
nx = uc:m_to_lu(4.8) + 1
ny = uc:m_to_lu(6.0) + 1
nz = uc:m_to_lu(2.8) + 1

-- viscosity
nu = uc:Nu_to_lu(1.511e-5)

-- Smagorinsky constant
C = 0.02
-- thermal diffusivity
nuT = 1.0e-2
--nuT = uc:Nu_to_lu(2.1e-4)
--nuT = 2.57e-5

-- Prandtl number
Pr = 0.713
-- Turbulent Prandtl number
Pr_t = 0.9

-- gravity * thermal expansion
gBetta = uc:gBetta_to_lu(3.0e-6)

-- initial temperature
Tinit = uc:Temp_to_lu(16 + 5.6216 / 2)
-- reference temperature
Tref = Tinit

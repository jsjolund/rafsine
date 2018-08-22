package.path = package.path .. ";lua/?.lua"
require "unitConverter"
-- Program Settings --
server_mode = false
if server_mode then
    FPS = 10
else
    FPS = 25
end

-- use openGL graphics
use_graphics = true
if server_mode then
    use_graphics = true
end

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
--print("Viscosity in LBM : ", nu)
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

-- Program Settings --

-- save the velocity/temperature to the hard-drive
save_fields = false
-- number of time-steps between two saves
save_dt = uc:s_to_N(1)

-- average the velocity/temperature fields
average_fields = false
-- start averaging when 'start_averaging' time-steps is reached
start_averaging = uc:s_to_N(2 * 60)
-- exit the program when 'stop' time-steps is reached
stop = uc:s_to_N(5 * 60)

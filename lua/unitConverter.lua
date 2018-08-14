-- uses the penlight library for its class
require "pl"
utils.import "pl.class"

UnitConverter = class()
function UnitConverter:_init(parameters)
  -- reference length in meters
  self.ref_L_phys = parameters["reference_length_in_meters"]
  -- reference length in number of nodes
  self.ref_L_lbm = parameters["reference_length_in_number_of_nodes"]

  -- reference speed in meter/second
  self.ref_U_phys = parameters["reference_speed_in_meters_per_second"]
  -- reference speed in lattice units (linked to the Mach number)
  self.ref_U_lbm = parameters["reference_speed_in_lattice_units"]

  -- temperature convertion factor
  self.C_Temp = parameters["temperature_convertion_factor"]
  -- reference temperature for Boussinesq in degres Celsius
  self.T0_phys = parameters["reference_temperature_physical"]
  self.T0_lbm = parameters["reference_temperature_LBM"]

  -- length convertion factor
  self.C_L = self.ref_L_phys / (self.ref_L_lbm - 1)
  --[[explanations on the lenght convertion factor:
  --  designed for C/C++ index standard
  --     [0; ...; N-1] --> total of N nodes
  --     Xmin <-> 0
  --     Xmax <-> N-1
  --  C/C++ memory is allocated with array[N]
  --                                 array[m_to_lu(Xmax)+1]
  --
  --  other possibilty: C_L = L_m / L_n
  --                    [0; ...;N] --> total N+1 nodes
  --                    Xmin <-> 0
  --                    Xmax <-> N
  --                    memory allocated with array[N+1]
  --                                          array[m_to_lu(Xmax)+1]
  --]]
  -- speed conversion factor
  self.C_U = self.ref_U_phys / self.ref_U_lbm

  -- time conversion factor
  self.C_T = self.C_L / self.C_U
end

-- round a floating point value to the closest integer
function UnitConverter:round(number)
  return math.floor(number + 0.5)
end

-- convert a distance in meters to a number of node (lattice unit)
function UnitConverter:m_to_lu(L_phys)
  if type(L_phys) == "number" then
    return self:round(L_phys / self.C_L)
  elseif type(L_phys) == "table" then
    local res = {}
    for i, Li in pairs(L_phys) do
      res[i] = self:m_to_lu(Li)
    end
    return res
  end
end

-- function to convert a position in real units
-- to a node-based position in lua
-- (shifted by 1 compared to C++)
function UnitConverter:m_to_LUA(L_phys)
  local L_n = self:m_to_lu(L_phys)
  if type(L_phys) == "number" then
    return 1 + L_n
  elseif type(L_phys) == "table" then
    return {1 + L_n[1], 1 + L_n[2], 1 + L_n[3]}
  end
end

-- function to convert a speed in meters/second to lattice units
function UnitConverter:ms_to_lu(U_phys)
  return U_phys / self.C_U
end

-- function to convert a volume flow rate in meters^3 / second
-- to a velocity in lattice unit
function UnitConverter:Q_to_Ulu(Q_phys, A_phys)
  return Q_phys / (self.C_U * A_phys)
end

-- function to convert the kinematic viscosity in meters^2 / second to lattice units
function UnitConverter:Nu_to_lu(Nu_phys)
  return Nu_phys / (self.C_U * self.C_L)
end

-- function to compute the relaxation time from the kinematic viscosity in meters^2 / second
function UnitConverter:Nu_to_tau(Nu_phys)
  return 0.5 + 3 * self:Nu_to_lu(Nu_phys)
end

-- function to compute the time convertion factor, i.e. the duration of one time-step ( in seconds)
function UnitConverter:N_to_s(nbr_iter)
  return self.C_T * nbr_iter
end

-- convert seconds to number of time-steps
function UnitConverter:s_to_N(seconds)
  return self:round(seconds / self.C_T)
end

-- convert physical temperature in Celsius to lbm temperature in lattice units
function UnitConverter:Temp_to_lu(Temp_phys)
  return self.T0_lbm + 1 / self.C_Temp * (Temp_phys - self.T0_phys)
end

-- convert g*Betta, i.e., gravity acceleration * coefficient of thermal expansion to lattice units
function UnitConverter:gBetta_to_lu(gBetta_phys)
  return gBetta_phys * self.C_T ^ 2 * self.C_Temp / self.C_L
end

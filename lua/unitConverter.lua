-- uses the penlight library for its class
require "pl"
utils.import "pl.class"

UnitConverter = class()
function UnitConverter:_init(parameters)
  -- reference length in meters
  local ref_L_phys = parameters["reference_length_in_meters"]
  -- reference length in number of nodes
  local ref_L_lbm = parameters["reference_length_in_number_of_nodes"]

  -- reference speed in meter/second
  local ref_U_phys = parameters["reference_speed_in_meters_per_second"]
  -- reference speed in lattice units (linked to the Mach number)
  local ref_U_lbm = parameters["reference_speed_in_lattice_units"]

  -- temperature convertion factor
  local C_Temp = parameters["temperature_convertion_factor"]
  -- reference temperature for Boussinesq in degres Celsius
  local T0_phys = parameters["reference_temperature_physical"]
  local T0_lbm = parameters["reference_temperature_LBM"]
  ucAdapter:set(ref_L_phys,ref_L_lbm,ref_U_phys,ref_U_lbm,C_Temp,T0_phys,T0_lbm)
end

function UnitConverter:C_U()
  return ucAdapter:C_U()
end

function UnitConverter:C_L()
  return ucAdapter:C_L()
end

function UnitConverter:C_T()
  return ucAdapter:C_T()
end

function UnitConverter:round(x)
  return ucAdapter:round(x)
end

function UnitConverter:m_to_lu(x)
  if type(x) == "number" then
    return ucAdapter:m_to_lu(x)
  elseif type(x) == "table" then
    local res = {}
    for i, Li in pairs(x) do
      res[i] = ucAdapter:m_to_lu(x)
    end
    return res
  end
end

function UnitConverter:m_to_LUA(x)
  if type(x) == "number" then
    return ucAdapter:m_to_LUA(x)
  elseif type(x) == "table" then
    local res = {}
    for i, Li in pairs(x) do
      res[i] = ucAdapter:m_to_LUA(x)
    end
    return res
  end
end

function UnitConverter:ms_to_lu(x)
  return ucAdapter:ms_to_lu(x)
end

function UnitConverter:Q_to_Ulu(q,a)
  return ucAdapter:Q_to_Ulu(q,a)
end

function UnitConverter:Nu_to_lu(x)
  return ucAdapter:Nu_to_lu(x)
end

function UnitConverter:Nu_to_tau(x)
  return ucAdapter:Nu_to_tau(x)
end

function UnitConverter:N_to_s(x)
  return ucAdapter:N_to_s(x)
end

function UnitConverter:s_to_N(x)
  return ucAdapter:s_to_N(x)
end

function UnitConverter:Temp_to_lu(x)
  return ucAdapter:Temp_to_lu(x)
end

function UnitConverter:gBetta_to_lu(x)
  return ucAdapter:gBetta_to_lu(x)
end
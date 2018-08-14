-- uses the penlight library for its class
require 'pl'
utils.import 'pl.class'
require "operators"
require "helpers"
require "settings"

max_bc_id = 0;

warning = [[
/*
 * Auto-generated file. Do not change this. Use the Lua code generator.
 */
]]

local function sgn(x)
  return x > 0 and 1 or x < 0 and -1 or 0
end

-- Class which calculates the voxel array indices of an axis aligned plane
VoxelPlane = class()
function VoxelPlane:_init(
name, -- Name of plane
origin, -- Origin in world coordinates (meters)
offset, -- Offset from origin (LUA)
dir1, -- Direction and length of the plane in meters (axis aligned vector)
dir2, -- Direction and length of the plane in meters (axis aligned vector)
normal) -- Plane normal

  -- Origin + normal in LUA coordinates. Places the BoundaryCondition one voxel in front of the inlet.
  -- Direction vectors in LUA coordinates
  local dir1_lua = vector(uc:m_to_LUA(vector(origin) + vector(dir1))) 
    - vector(uc:m_to_LUA(vector(origin))) + floor(vector(dir1)/norm(vector(dir1)))
  local dir2_lua = vector(uc:m_to_LUA(vector(origin) + vector(dir2))) 
    - vector(uc:m_to_LUA(vector(origin))) + floor(vector(dir2)/norm(vector(dir2)))
  self.name = name
  self.origin = uc:m_to_LUA(vector(origin))
  self.n1 = math.abs(dir1_lua[1]) + math.abs(dir1_lua[2]) + math.abs(dir1_lua[3])
  self.n2 = math.abs(dir2_lua[1]) + math.abs(dir2_lua[2]) + math.abs(dir2_lua[3])
  self.d1 = { sgn(dir1_lua[1]), sgn(dir1_lua[2]), sgn(dir1_lua[3]) }
  self.d2 = { sgn(dir2_lua[1]), sgn(dir2_lua[2]), sgn(dir2_lua[3]) }
  self.normal = normal
  self.indices = {}
  -- Calculate indices of voxels starting at the origin, in order of dir1
  for i = 0, self.n1 - 1, 1 do
    for j = 0, self.n2 - 1, 1 do
      local x = self.origin[1] + offset[1] + self.d1[1] * i + self.d2[1] * j
      local y = self.origin[2] + offset[2] + self.d1[2] * i + self.d2[2] * j
      local z = self.origin[3] + offset[3] + self.d1[3] * i + self.d2[3] * j
      local index = x + y * nx + z * nx * ny
      self.indices[#self.indices + 1] = index
    end
  end
  self.origin = uc:m_to_LUA(vector(origin) + vector(offset) * C_L)
end

-- C++ code generator for the plane
function VoxelPlane:to_c_init(struct_name, indices_name)
  local indices_init = "const int "
      .. indices_name .. "[" .. self.n1 * self.n2 .. "]="
      .. "{" .. table.concat(self.indices, ",") .. "};"

  local struct_init =
  "VoxelPlane " .. struct_name .. "={" ..
      ".name=\"" .. self.name .. "\"" .. "," ..
      ".origin={" .. table.concat(self.origin, ",") .. "}," ..
      ".normal={" .. table.concat(self.normal, ",") .. "}," ..
      ".n1=" .. self.n1 .. "," ..
      ".n2=" .. self.n2 .. "," ..
      ".d1={" .. table.concat(self.d1, ",") .. "}," ..
      ".d2={" .. table.concat(self.d2, ",") .. "}," ..
      ".indices=&" .. indices_name .. "[0]" ..
      "};"
  return indices_init .. "\n" .. struct_init
end

-- Constructor for boundary condition type
require "NodeDescriptor"
require "NodeD3Q6"
BoundaryConditionData = class()
function BoundaryConditionData:_init(
id, -- Voxel id of this boundary condition
normal, -- Normal
velocity, -- Velocity (axis aligned vector)
temperature, -- Temperature
rel_pos, -- Relative position
typeT, -- Temperature type
typeBC) -- Boundary condition type

  if id > max_bc_id then max_bc_id = id end
  self.id = id
  self.t = temperature
  if temperature == nil then self.t = 0 else self.t = temperature end
  self.v = vector(velocity)
  self.n = vector(normal)
  if rel_pos == nil then self.rel_pos_len = 0 else self.rel_pos_len = rel_pos end
  if typeBC == "inlet" then self.typeBC = typeBC .. "_" .. typeT else self.typeBC = typeBC end
  self.rel_pos = -(1 + uc:m_to_lu(self.rel_pos_len)) * self.n
end

-- C++ code generation for boundary condition
function BoundaryConditionData:to_c_init(struct_name)
  local struct_init =
  "BoundaryConditionData " .. struct_name .. "={" ..
      ".id=" .. self.id .. "," ..
      ".t=" .. self.t .. "," ..
      ".v={" .. table.concat(self.v, ",") .. "}," ..
      ".n={" .. table.concat(self.n, ",") .. "}," ..
      ".rel_pos={" .. table.concat(self.rel_pos, ",") .. "}," ..
      ".typeBC=" .. self.typeBC .. "};"
  return struct_init
end

require "helpers"
require "settings"
-- Class which calcualtes the voxel array indices of a pair of inlet and outlet
BoundaryCondition = class()
function BoundaryCondition:_init(inletID, inlet, outletID, outlet)

  local inlet = vector(inlet)
  local outlet = vector(outlet)

  self.name = inlet.parentName
  self.tCSVheader = inlet.tCSVheader
  self.qCSVheader = inlet.qCSVheader
  self.tOutAvgCSVheaders = outlet.tAvgCSVheaders;
  self.tInAvgCSVheaders = inlet.tAvgCSVheaders;
  self.qOutAvgCSVheaders = outlet.qAvgCSVheaders;
  self.qInAvgCSVheaders = inlet.qAvgCSVheaders;

  -- Calculate the midpoint of the plane in LUA coordinates
  self.midpoint = vector(uc:m_to_LUA(inlet.origin + vector(inlet.dir1)*0.5 + vector(inlet.dir2)*0.5))

  self.inletPlane = VoxelPlane(inlet.name, inlet.origin, inlet.normal, inlet.dir1, inlet.dir2, inlet.normal)
  self.outletPlane = VoxelPlane(outlet.name, outlet.origin, outlet.normal, outlet.dir1, outlet.dir2, outlet.normal)

  local t = vector(inlet.temperature)
  local v = vector(inlet.velocity)
  local n = vector(inlet.normal)
  self.inletBC = BoundaryConditionData(inletID, n, v, t.value, t.rel_pos, t.type_, inlet.typeBC)

  local t = vector(outlet.temperature)
  local v = vector(outlet.velocity)
  local n = vector(outlet.normal)
  self.outletBC = BoundaryConditionData(outletID, n, v, t.value, t.rel_pos, t.type_, outlet.typeBC)

  if #outlet.tAvgCSVheaders ~= #inlet.tAvgCSVheaders then
    error("Number of inlet and outlet averages must be equal.")
  end
  self.num_averages = #outlet.tAvgCSVheaders
  return self
end

-- C++ code generation for pair of inlet and outlet
function BoundaryCondition:to_c_init()
  local inletBC_varname = self.inletPlane.name .. "Data"
  local inletBC = self.inletBC:to_c_init(inletBC_varname)

  local outletBC_varname = self.outletPlane.name .. "Data"
  local outletBC = self.outletBC:to_c_init(outletBC_varname)

  local inletPlane_varname = self.inletPlane.name
  local inletPlane = self.inletPlane:to_c_init(inletPlane_varname, inletPlane_varname .. "Indices")

  local outletPlane_varname = self.outletPlane.name
  local outletPlane = self.outletPlane:to_c_init(outletPlane_varname, outletPlane_varname .. "Indices")
  
  local avg_temp_in = {}
  for i = 1, self.num_averages do avg_temp_in[i] = "0.0f" end
  local avg_temp_in_varname = self.name .. "avgTempIn"
  local avg_temp_in = "float " .. avg_temp_in_varname .. "[" .. self.num_averages .. "]={" .. table.concat(avg_temp_in, ",") .. "};"
  
  local avg_temp_out = {}
  for i = 1, self.num_averages do avg_temp_out[i] = "0.0f" end
  local avg_temp_out_varname = self.name .. "avgTempOut"
  local avg_temp_out = "float " .. avg_temp_out_varname .. "[" .. self.num_averages .. "]={" .. table.concat(avg_temp_out, ",") .. "};"
  
  local avg_temp_delta = {}
  for i = 1, self.num_averages do avg_temp_delta[i] = "0.0f" end
  local avg_temp_delta_varname = self.name .. "avgTempDelta"
  local avg_temp_delta = "float " .. avg_temp_delta_varname .. "[" .. self.num_averages .. "]={" .. table.concat(avg_temp_delta, ",") .. "};"

  local avg_vel_in = {}
  for i = 1, self.num_averages do avg_vel_in[i] = "0.0f" end
  local avg_vel_in_varname = self.name .. "avgVelIn"
  local avg_vel_in = "float " .. avg_vel_in_varname .. "[" .. self.num_averages .. "]={" .. table.concat(avg_vel_in, ",") .. "};"
  
  local avg_vel_out = {}
  for i = 1, self.num_averages do avg_vel_out[i] = "0.0f" end
  local avg_vel_out_varname = self.name .. "avgVelOut"
  local avg_vel_out = "float " .. avg_vel_out_varname .. "[" .. self.num_averages .. "]={" .. table.concat(avg_vel_out, ",") .. "};"
  
  local avg_vel_delta = {}
  for i = 1, self.num_averages do avg_vel_delta[i] = "0.0f" end
  local avg_vel_delta_varname = self.name .. "avgVelDelta"
  local avg_vel_delta = "float " .. avg_vel_delta_varname .. "[" .. self.num_averages .. "]={" .. table.concat(avg_vel_delta, ",") .. "};"

  local avg_q_in = {}
  for i = 1, self.num_averages do avg_q_in[i] = "0.0f" end
  local avg_q_in_varname = self.name .. "avgQIn"
  local avg_q_in = "float " .. avg_q_in_varname .. "[" .. self.num_averages .. "]={" .. table.concat(avg_q_in, ",") .. "};"
  
  local avg_q_out = {}
  for i = 1, self.num_averages do avg_q_out[i] = "0.0f" end
  local avg_q_out_varname = self.name .. "avgQOut"
  local avg_q_out = "float " .. avg_q_out_varname .. "[" .. self.num_averages .. "]={" .. table.concat(avg_q_out, ",") .. "};"
  
  local avg_q_delta = {}
  for i = 1, self.num_averages do avg_q_delta[i] = "0.0f" end
  local avg_q_delta_varname = self.name .. "avgQDelta"
  local avg_q_delta = "float " .. avg_q_delta_varname .. "[" .. self.num_averages .. "]={" .. table.concat(avg_q_delta, ",") .. "};"

  local avg_den_in = {}
  for i = 1, self.num_averages do avg_den_in[i] = "0.0f" end
  local avg_den_in_varname = self.name .. "avgDenIn"
  local avg_den_in = "float " .. avg_den_in_varname .. "[" .. self.num_averages .. "]={" .. table.concat(avg_den_in, ",") .. "};"
  
  local avg_den_out = {}
  for i = 1, self.num_averages do avg_den_out[i] = "0.0f" end
  local avg_den_out_varname = self.name .. "avgDenOut"
  local avg_den_out = "float " .. avg_den_out_varname .. "[" .. self.num_averages .. "]={" .. table.concat(avg_den_out, ",") .. "};"
  
  local avg_den_delta = {}
  for i = 1, self.num_averages do avg_den_delta[i] = "0.0f" end
  local avg_den_delta_varname = self.name .. "avgDenDelta"
  local avg_den_delta = "float " .. avg_den_delta_varname .. "[" .. self.num_averages .. "]={" .. table.concat(avg_den_delta, ",") .. "};"

  local bc = "BoundaryConditions " .. self.name .. "={" ..
      ".name=\"" .. self.name .. "\"" .. "," ..
      ".tCSVheader=\"" .. self.tCSVheader .. "\"" .. "," ..
      ".qCSVheader=\"" .. self.qCSVheader .. "\"" .. "," ..
      ".csvT=0.0," .. 
      ".csvQ=0.0," ..
      ".midpoint={" .. table.concat(self.midpoint, ",") .. "}" .. "," ..
      ".inlet=&" .. inletBC_varname .. "," ..
      ".outlet=&" .. outletBC_varname .. "," ..
      ".inletPlane=&" .. inletPlane_varname .. "," ..
      ".outletPlane=&" .. outletPlane_varname .. "," ..
      ".num_avg=" .. self.num_averages .. "," ..
      ".avg_temp_in=&" .. avg_temp_in_varname .. "[0]," ..
      ".avg_temp_out=&" .. avg_temp_out_varname .. "[0]," ..
      ".avg_temp_delta=&" .. avg_temp_delta_varname .. "[0]," ..
      ".avg_vel_in=&" .. avg_vel_in_varname .. "[0]," ..
      ".avg_vel_out=&" .. avg_vel_out_varname .. "[0]," ..
      ".avg_vel_delta=&" .. avg_vel_delta_varname .. "[0]," ..
      ".avg_q_in=&" .. avg_q_in_varname .. "[0]," ..
      ".avg_q_out=&" .. avg_q_out_varname .. "[0]," ..
      ".avg_q_delta=&" .. avg_q_delta_varname .. "[0]," ..
      ".avg_den_in=&" .. avg_den_in_varname .. "[0]," ..
      ".avg_den_out=&" .. avg_den_out_varname .. "[0]," ..
      ".avg_den_delta=&" .. avg_den_delta_varname .. "[0]," ..
      "};"
  return inletBC .. "\n" .. outletBC .. "\n" 
      .. inletPlane .. "\n" .. outletPlane .. "\n" 
      .. avg_temp_in .. "\n" .. avg_temp_out .. "\n" .. avg_temp_delta .. "\n" 
      .. avg_vel_in .. "\n" .. avg_vel_out .. "\n" .. avg_vel_delta .. "\n" 
      .. avg_q_in .. "\n" .. avg_q_out .. "\n" .. avg_q_delta .. "\n" 
      .. avg_den_in .. "\n" .. avg_den_out .. "\n" .. avg_den_delta .. "\n" 
      .. bc .. "\n"
end

BoundaryConditions = class()
function BoundaryConditions:_init()
  self.bcs = {}
end

function BoundaryConditions:add(boundaryCondition)
  self.bcs[#self.bcs + 1] = boundaryCondition
end

function BoundaryConditions:header_start()
  return warning ..
[[
#ifndef BCS_H
#define BCS_H
#include "../src/BoundaryCondition.hpp"
]]
end

function BoundaryConditions:header_end()
  return "\n\n#endif"
end

function BoundaryConditions:source(headerName)
  return warning .. '#include \"' .. headerName .. '\"\n'
end

function BoundaryConditions:saveToFile(fileName)
  -- Start of source file
  fc = io.open(fileName .. ".cpp", "w")
  fc:write(self:source(fileName .. ".hpp"))
  for i, s in pairs(self.bcs) do
    fc:write(s:to_c_init())
  end

  -- Add boundary condition array
  table.sort(self.bcs, function(a, b) return a.name < b.name end)
  fc:write("\nstd::array<BoundaryConditions*, NUM_BCS> boundaryConditions={")
  for i, s in pairs(self.bcs) do
    fc:write("&" .. s.name .. ", ")
  end
  fc:write("};")
  fc:write("\nBoundaryConditionData emptyBC = {.id = 0, .t = 0, .v = {0, 0, 0}, .n = {0, 0, 0}, .rel_pos = {-0, -0, -0}, .typeBC = none};")
  fc:close()
  -- End of source file

  -- Start of header file
  fh = io.open(fileName .. ".hpp", "w")
  fh:write(self.header_start())

  -- Number of boundary conditions
  fh:write("\n#define NUM_BCS " .. #self.bcs)
  fh:write("\n#define NUM_BCTYPES " .. (max_bc_id + 1))
  fh:write("\n#define CSV_INPUT_HEADERS \"time0\", \"time1\"")
  for i, s in pairs(self.bcs) do
    fh:write(", \"" .. s.tCSVheader .. "\", \"" .. s.qCSVheader .. "\"")
  end

  fh:write("\n#define CSV_INPUT_VARS ")
  for i, s in pairs(self.bcs) do
    fh:write(s.name .. ".csvT, " .. s.name .. ".csvQ")
    if i ~= #self.bcs then
      fh:write(", ")
    end
  end

  fh:write("\n#define CSV_OUTPUT_HEADERS \"time\"")
  for i, s in pairs(self.bcs) do
    fh:write(", \"" 
    .. table.concat(s.tInAvgCSVheaders, "\", \"") 
    .. "\", \"" 
    .. table.concat(s.tOutAvgCSVheaders, "\", \"")
    .. "\", \"" 
    .. table.concat(s.qInAvgCSVheaders, "\", \"")
    .. "\", \"" 
    .. table.concat(s.qOutAvgCSVheaders, "\", \"")
     .. "\"")
  end
  
  csvNumOutputHeaders = 1
  for i, s in pairs(self.bcs) do
    csvNumOutputHeaders = csvNumOutputHeaders + #s.tInAvgCSVheaders + #s.tOutAvgCSVheaders + #s.qInAvgCSVheaders + #s.qOutAvgCSVheaders
  end
  fh:write("\n#define CSV_NUM_OUTPUT_HEADERS " .. csvNumOutputHeaders)
  
  fh:write("\n")

  -- Extern arrays
  fh:write("\nextern BoundaryConditionData emptyBC;")
  fh:write("\nextern std::array<BoundaryConditions*, NUM_BCS> boundaryConditions;")
  fh:write("\nextern BoundaryConditions ")
  for i, s in pairs(self.bcs) do
    fh:write(s.name)
    if i ~= #self.bcs then
      fh:write(", ")
    end
  end
  fh:write(";")

  fh:write(self.header_end())
  fh:close()
  -- End of header file
end

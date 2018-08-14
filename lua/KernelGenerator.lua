require "ThreadLayout"
require "MemoryManager"
require "ModelParameters"
require "DynamicsGenerator"

-- Generate the CUDA kernel for LBM

KernelGenerator = class()
function KernelGenerator:_init(kernelName, fileName)
  -- name of the kernel
  self.name = kernelName or "ComputeKernel"
  -- name of the file where the kernel is generated
  self.fileName = fileName or "kernel.h"

  -- check destination folder exist
  checkFolder(src_destination_folder)
  -- file writter for kernel.h
  self.filewriter = FileWriter(src_destination_folder .. "/" .. self.fileName)
  -- short version
  self.f = self.filewriter
  self.w = function(text)
    if type(text) == "string" then
      self.filewriter:writeLine(text)
    elseif type(text) == "table" then
      for _, line in pairs(text) do
        self.w(line)
      end
    end
  end
end

-- generate the CUDA kernel into the file 'fileName'
-- and returns "#include 'fileName' "
function KernelGenerator:generate()
  self:genMacros()
  self.f:startKernel(self:genKernelDef())
  self:genLocalVars()
  self:genThreadIndex()
  --TODO : proper boundary conditions
  --self:genStreaming()
  self:genStreamingPeriodic()
  self:genMomentsComputation()
  -- store some of the moments if required
  if main.storeVelocity then -- store velocity
    for i, v in ipairs(model:genVelocity()) do
      self.w(memory:genAccess_GPU(model.velocity_name[i]:upper(), model.position_name) .. " = " .. v)
    end
  end
  if main.storeDensity then
    self.w(memory:genAccess_GPU(model.density_name:upper(), model.position_name) .. " = " .. model.density_name)
  end
  --[[
    self.f:conditionalBlock(
      "(x-LatSizeX/2)*(x-LatSizeX/2)+(y-LatSizeY/2)*(y-LatSizeY/2) < 100",
      { "vx = 0.1f", "vy = 0" }
    )
    --]]
  --Add the code generated from the modules
  for _, m in pairs(main.modules) do
    if m.genKernelCode then
      self.w(m:genKernelCode())
    end
  end
  self:genEquilibriumComputation()
  self:genRelaxation()
  self.f:endFunction()
end

-- generate the macros needed by the kernel
function KernelGenerator:genMacros()
  self.filewriter:startMacros("macros needed by the kernel")
  self.w(memory:genMacros())
  self.filewriter:endMacros()
end

-- generate the CUDA kernel definition
-- TODO: store the arguments of kernel call and kernel definition in a single table
--       to avoid possible mismatch
function KernelGenerator:genKernelDef()
  args = {}
  table.insert(args, memory:genFuncDefArgs())
  table.insert(args, model:genFuncDefArgs())
  -- look if some modules need to pass arguments to the kernel
  for _, m in pairs(main.modules) do
    if m.genKernelDefArgs then
      table.insert(args, m:genKernelDefArgs())
    end
  end
  -- if the velocity need to be stored, add it as an argument
  if main.storeVelocity then
    for _, v in pairs(model.velocity_name) do
      table.insert(args, model.precision .. "* " .. v:upper())
    end
  end
  -- if the density need to be stored, add it as an argument
  if main.storeDensity then
    table.insert(args, model.precision .. "* " .. model.density_name:upper())
  end
  args = table.concat(args, ", ")
  return "void " .. self.name .. "(" .. args .. ")"
end

-- generate the call to the CUDA kernel from C
-- args are additional arguments (optional)
function KernelGenerator:genCall(args)
  local result = ""
  result = result .. self.name
  result = result .. "<<<" .. layout.gridName .. ", " .. layout.blockName .. ">>>"
  result = result .. "(" .. memory:genFuncCallArgs() .. ", " .. model:genFuncCallArgs()
  if args ~= "" then
    result = result .. ", " .. args
  end
  result = result .. ")"
  return result
end

-- generate how to compute indices
function KernelGenerator:genThreadIndex()
  local X = model.position_name
  self.f:commentSection("Compute thread index")
  self.f:comment("position of the node")
  self.w(layout:genNodeDefList(X))
  if (layout.needNodePositionChecking) then
    local conditions = merge({X, ">=", model.size_names})
    conditions = table.concat(conditions, " || ")
    self.f:comment("check that the node is inside the domain")
    self.f:conditionalBlock(conditions, "return")
  end
end

--generate the declaration of the registers used by the kernel
function KernelGenerator:genLocalVars()
  self.f:comment("Store the distribution functions in registers")
  self.w(model.precision .. " " .. node:genAllDistributionNames())
  if model.genLocalVarsDef then
    self.w(model:genLocalVarsDef())
  end
end

--generate the streaming of the distribution functions
function KernelGenerator:genStreaming()
  self.f:commentSection("Streaming step")
  --self.w("if( (x==0) || (x==LatSizeX-1) || (y==0) || (y==LatSizeY-1) || (z==0) || (z==LatSizeZ-1) ) return")
  self.w("if( (x==0) || (x==LatSizeX-1) || (y==0) || (y==LatSizeY-1) ) return")
  self:genStreamingBulk()
end

-- generate the streaming for the bulk of the domain (inside of the boundaries)
function KernelGenerator:genStreamingBulk()
  local X = model.position_name
  for i = 1, node.Q do
    local fi = node:genDistributionName(i)
    local ei = node:getDirection(i)
    self.w(fi .. " = " .. f(i, X - ei))
  end
end

-- Function to generate a streaming with periodic boundary conditions
function KernelGenerator:genStreamingPeriodic()
  self.f:commentSection("Streaming step (periodic)")
  local X = model.position_name
  -- generate periodic neighbours
  for i, x in pairs(X) do
    local xmax = model.size_names[i] - 1
    self.w("int " .. x .. sign(1) .. " = " .. ternary(x .. "==" .. xmax, 0, x + 1))
    self.w("int " .. x .. sign(-1) .. " = " .. ternary(x .. "==" .. "0", xmax, x - 1))
  end
  -- stream using the periodic neighbours
  for i = 1, node.Q do
    local fi = node:genDistributionName(i)
    local ei_sign = sign(-node:getDirection(i))
    self.w(fi .. " = " .. f(i, merge({X, ei_sign})))
  end
end

-- generate the computation of the moments required by the model
function KernelGenerator:genMomentsComputation()
  self.f:commentSection("Compute moments")
  if (model.matrix_transformation) then
    -- compute all the moments using the matrix transformation
    local moments = model.M * node:genDFNamesMatrix()
    for i = 1, node.Q do
      self.w(model.precision .. " " .. model.moments_names[i][1] .. " = " .. moments[i][1])
    end
  else
    -- compute each moment from its formula
    for _, m in pairs(dynamics.moments) do
      self.f:comment(m.description)
      self.w(m:genComputations())
    end
  end
end

-- Generate the computation of the equilibrium dfs
function KernelGenerator:genEquilibriumComputation()
  if model.comput_space == "dist_func" then
    self.f:commentSection("Compute Equilibrium Distribution Function")
    for i, fi_eq in ipairs(dynamics:genEquilibriumDfs()) do
      self.w(node:genEquilibriumDistributionName(i) .. " = " .. fi_eq)
    end
  elseif model.comput_space == "moment" then
    if dynamics.genEquilibriumMoments then
      self.f:commentSection("Compute Equilibrium Moments")
      self.w(dynamics:genEquilibriumMoments())
    end
  end
end

function KernelGenerator:genRelaxation()
  local X = model.position_name
  self.f:commentSection("Relax Toward Equilibrium")
  if model.comput_space == "dist_func" then
    if dynamics.genRelaxation then
      self.w(dynamics.genRelaxation())
    else
      for i, fi in ipairs(dynamics:genBGK()) do
        self.w(memory:f_tmp(i, X) .. " = " .. fi)
      end
    end
  elseif model.comput_space == "moment" then
    self.w(dynamics:genRelaxationMoments())
    if (model.matrix_transformation) then
      -- compute distribution functions using the inverse transformation matrix
      local fis_mat = model.invM * model.moments_names
      for i = 1, node.Q do
        self.w(memory:f_tmp(i, X) .. " = " .. fis_mat[i][1])
      end
    else
      for i, fi in ipairs(dynamics:genDFComputations()) do
        self.w(memory:f_tmp(i, X) .. " = " .. fi)
      end
    end
  end
end

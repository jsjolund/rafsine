-- Handles the memory in C and CUDA

MemoryManager = class()
function MemoryManager:_init()
  -- name used for the distribution functions
  self.df = "df"
  -- suffix used for temporary arrays
  self.suffix = "_tmp"
  -- temporary lattice TODO: a df class that generate temp?
  self.df_tmp = self.df .. self.suffix
end

-- Generate the memory allocation of a array on the GPU
-- inputs:
--   name   name of the variable to allocate
--   type_  type of the objects stored in the array
--   size   number of elements in the array
function MemoryManager:genMemAllocGPU(name_, type_, size_)
  local type_ = type_ or model.precision
  local size_ = size_ or PROD(model.size_names)
  return "thrust::device_vector<" .. type_ .. "> " .. name_ .. "(" .. size_ .. ")"
end

-- Generate the memory allocation of a array on the CPU
function MemoryManager:genMemAllocCPU(name_, type_, size_)
  local type_ = type_ or model.precision
  local size_ = size_ or PROD(model.size_names)
  return "thrust::host_vector<" .. type_ .. "> " .. name_ .. "(" .. size_ .. ")"
end

-- Generate a pointer to the CPU/GPU memory for the variable var
function MemoryManager:genPointer(var)
  return "thrust::raw_pointer_cast(&(" .. var .. ")[0])"
end

-- Generate the memory allocation of the distribution functions
function MemoryManager:genMemAllocDfs(suffix)
  local suffix = suffix or ""
  return "DistributionFunctionsGroup " ..
    self.df .. suffix .. "(" .. node.Q .. "," .. table.concat(model.size_names, ", ") .. ")"
end

-- Generate the memory allocation of the temporary distribution functions
function MemoryManager:genMemAllocDfs_tmp()
  return self:genMemAllocDfs(self.suffix)
end

-- generate the arguments for a function definition
function MemoryManager:genFuncDefArgs()
  return model.precision .. " *" .. self.df .. ", " .. model.precision .. " *" .. self.df_tmp
end

-- generate the argument for a function call
function MemoryManager:genFuncCallArgs()
  -- how to access gpu memory from a DistributionFunctionsGroup
  local ptr = ".gpu_ptr()"
  return self.df .. ptr .. ", " .. self.df_tmp .. ptr
end

-- generate the macros needed to compute the memory indices
-- TODO: there could be 2 options:
--       1) define macros and use macros in the kernel (currently implemented)
--       2) don't define macros and pre-expend indices computations with LUA
function MemoryManager:genMacros()
  if node.D == 2 then
    return {
      "#define I2D(x,y,nx,ny) ((x) + (y)*(nx))",
      "#define Idf2D(i,x,y,nx,ny) ((i)*(nx)*(ny) + (x) + (y)*(nx))",
      "#define df2D(    i,x,y,nx,ny)  (" .. self.df .. "[    Idf2D(i,x,y,nx,ny)])",
      "#define dftmp2D( i,x,y,nx,ny)  (" .. self.df_tmp .. "[Idf2D(i,x,y,nx,ny)])"
    }
  else
    return {
      "#define I3D(x,y,z,nx,ny,nz) ((x) + (y)*(nx) + (z)*(nx)*(ny))",
      "#define Idf3D(i,x,y,z,nx,ny,nz) ((i)*(nx)*(ny)*(nz) + (x) + (y)*(nx) + (z)*(nx)*(ny))",
      "#define df3D(    i,x,y,z,nx,ny,nz)  (" .. self.df .. "[    Idf3D(i,x,y,z,nx,ny,nz)])",
      "#define dftmp3D( i,x,y,z,nx,ny,nz)  (" .. self.df_tmp .. "[Idf3D(i,x,y,z,nx,ny,nz)])"
    }
  end
end

-- generate the computation of the memory index
function MemoryManager:genIdx(x, y, z)
  if type(x) == "table" then
    y = x[2]
    z = x[3]
    x = x[1] -- last so no overwrite
  end
  if node.D == 2 then
    return "I2D(" .. x .. "," .. y .. "," .. model.sizeX.name .. "," .. model.sizeY.name .. ")"
  else -- D == 3
    return "I3D(" ..
      x .. "," .. y .. "," .. z .. "," .. model.sizeX.name .. "," .. model.sizeY.name .. "," .. model.sizeZ.name .. ")"
  end
end

-- generate the declaration of the memory index
function MemoryManager:genIdxDef(idx, X)
  return "unsigned int " .. idx .. " = " .. self:genIdx(X[1], X[2], X[3])
end

-- Generate linear access to a simple GPU array
function MemoryManager:genAccess1D_GPU(var, index)
  return var .. "[" .. index .. "]"
end

-- Generate 2D/3D access to a simple GPU array
function MemoryManager:genAccess_GPU(var, x, y, z)
  return var .. "[" .. self:genIdx(x, y, z) .. "]"
end

-- Generate linear access to the CPU memory (of the ith df)
-- TODO: automatic detectiong of whether the call is made from a kernel or form the CPU
--       and deduce to call f_CPU or f_GPU
function MemoryManager:genAccessToDf_CPU(i, index)
  if type(index) == "string" then
    return self.df .. "(" .. (i - 1) .. "," .. index .. ")"
  elseif type(index) == "table" then
    return self.df .. "(" .. (i - 1) .. "," .. table.concat(index, ",") .. ")"
  else
    error("index has incorrect format")
  end
end

--[[
function MemoryManager:genAccessToDf_GPU(i, index)
  return self.df.."["..(i-1).."]["..index.."]"
end
function MemoryManager:genAccessToDf_tmp_GPU(i, index)
  return self.df_tmp.."["..(i-1).."]["..index.."]"
end
--]]
-- function to convert a table to 3 indices
function MemoryManager:triplet(x, y, z)
  if type(x) == "table" then
    return x[1], x[2], x[3]
  else
    return x, y, z
  end
end

--generate the code required to access the ith distribution function
function MemoryManager:genAccess(i, x, y, z)
  i = tostring(i - 1)
  x, y, z = self:triplet(x, y, z)
  local nx = model.sizeX.name
  local ny = model.sizeY.name
  local nz = model.sizeZ.name
  if node.D == 2 then
    return "df2D(" .. i .. ", " .. x .. "," .. y .. ", " .. nx .. "," .. ny .. ")"
  else -- D == 3
    return "df3D(" .. i .. ", " .. x .. "," .. y .. "," .. z .. ", " .. nx .. "," .. ny .. "," .. nz .. ")"
  end
end

--generate the code required to access the temporary ith distribution function
function MemoryManager:genAccessToTemp(i, x, y, z)
  i = tostring(i - 1)
  x, y, z = self:triplet(x, y, z)
  local nx = model.sizeX.name
  local ny = model.sizeY.name
  local nz = model.sizeZ.name
  if node.D == 2 then
    return "dftmp2D(" .. i .. ", " .. x .. "," .. y .. ", " .. nx .. "," .. ny .. ")"
  else -- D == 3
    return "dftmp3D(" .. i .. ", " .. x .. "," .. y .. "," .. z .. ", " .. nx .. "," .. ny .. "," .. nz .. ")"
  end
end

-- generate access to the ith df with X = { "x","y","z" }
function MemoryManager:f(i, X)
  return self:genAccess(i, X[1], X[2], X[3])
end

-- generate access to the temporary ith df with X = { "x","y","z" }
function MemoryManager:f_tmp(i, X)
  return self:genAccessToTemp(i, X[1], X[2], X[3])
end

-- useful wrapping of memory:f(i,X)
function f(i, X)
  return memory:f(i, X)
end
-- useful wrapping of memory:f_tmp(i,X)
function f_tmp(i, X)
  return memory:f_tmp(i, X)
end

-- Generate the upload of the dfs from the CPU to the GPU
function MemoryManager:genMemCopyCpuToGpu()
  return self.df .. ".upload()"
end

-- Swap distributions functions
function MemoryManager:genSwap()
  return "DistributionFunctionsGroup::swap(" .. self.df .. "," .. self.df_tmp .. ")"
end

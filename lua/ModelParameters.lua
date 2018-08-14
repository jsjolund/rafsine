require "pl"
utils.import "pl.class"
-- The Parameter class describe a parameter : type, name, value, ...
-- The BaseModel class regroups all the parameters as well as function to generate code related to them
-- The BGKModel derives the BaseModel for the parameters needed in the BGK LBM

-- Define a parameter structure
Parameter = class()
function Parameter:_init(typeC, name, value, description)
  -- State that this is a parameter
  self.className = "parameter"
  -- type of the parameter in the generated code
  self.typeC = typeC
  -- name of the parameter in the generated code
  self.name = name
  -- value of the parameter in lua (and in the generated code)
  self.value = value
  -- description of the parameter (optional but recommended)
  self.description = description or ""
end

-- generate the name to be used in the code
function Parameter:genName()
  return self.name
end

-- generate the full name: type + name (declaration)
function Parameter:genFullName()
  return self.typeC .. " " .. self.name
end

-- generate the declaration of the parameter
-- ex: unsigned int sizeX
function Parameter:genDeclaration()
  return self:genFullName()
end

-- generate the initialisation of the parameter
-- ex: sizeX = 64
function Parameter:genInitialisation()
  return self.name .. " = " .. self.value
end

-- generate the declaration-initialisation of the parameter
-- ex: unsigned int sizeX = 64
function Parameter:genDeclInit()
  return self:genDeclaration() .. " = " .. self.value
end

-- Define the model used as well as the parameters of the model
BaseModel = class()
function BaseModel:_init(name, precision, computational_space, sizeX, sizeY, sizeZ)
  -- name of the model
  self.name = name
  -- Defines in which space the computation are done.
  -- 'dist_func' : in the same space as the distribution functions (ex: BGK)
  -- 'moment'    : in the moment space (ex: MRT)
  self.comput_space = computational_space
  -- precision of the computation (float or double)
  if precision == "single_precision" then
    self.precision = "float"
  elseif precision == "double_precision" then
    self.precision = "double"
  else
    self.precision = "float"
  end
  -- apply the chosen precision to the node weights
  node:applyPrecision(self.precision)
  -- size of the lattice along x,y,z direction
  -- the parameters are stored in a table
  self.sizeX = Parameter("int", "LatSizeX", sizeX, "Size of the lattice along X axis")
  self.sizeY = Parameter("int", "LatSizeY", sizeY, "Size of the lattice along Y axis")
  self.sizeZ = Parameter("int", "LatSizeZ", sizeZ or 1, "Size of the lattice along Z axis")
  -- regroup the name of the sizes in a vector
  if node.D == 2 then
    self.size_names = vector({self.sizeX.name, self.sizeY.name})
  else
    self.size_names = vector({self.sizeX.name, self.sizeY.name, self.sizeZ.name})
  end

  -- NAMES OF A FEW PHYSICAL QUANTITIES
  -- name of the density
  self.density_name = "rho"
  --name of the components of the velocity
  if node.D == 2 then
    self.velocity_name = vector({"vx", "vy"})
  else
    self.velocity_name = vector({"vx", "vy", "vz"})
  end
  --name of the component of the node position
  if node.D == 2 then
    self.position_name = vector({"x", "y"})
  else
    self.position_name = vector({"x", "y", "z"})
  end
  -- name of the current node index in a kernel
  self.nodeIdx = "idx"
  -- name of the number of time-steps since the beginning of the simulation
  self.time_name = "time"
end

-- Generate the computation of the number of cells
-- (i.e., sizeX*sizeY*sizeZ)
function BaseModel:genNumberOfCells()
  if node.D == 2 then
    return self.sizeX.name .. "*" .. self.sizeY.name
  else --node.D == 3
    return self.sizeX.name .. "*" .. self.sizeY.name .. "*" .. self.sizeZ.name
  end
end

-- generate a list of the parameters
function BaseModel:genParametersList()
  local params = {}
  for _, p in pairs(self) do
    if (type(p) == "table" and p.className == "parameter") then -- Parameter class is seen as a table
      if (node.D == 3 or p ~= self.sizeZ) then -- don't declare sizeZ if the node is 2D
        table.insert(params, p)
      end
    end
  end
  return params
end

-- generate a string of the parameters (to be used as argument of functions)
-- if mode=="declare" then the string will include types
function BaseModel:genArguments(mode)
  local args = {}
  local idx = 1
  for i, p in pairs(self) do
    if (type(p) == "table" and p.className == "parameter") then -- Parameter class is seen as a table
      if (node.D == 3 or p ~= self.sizeZ) then -- don't declare sizeZ if the node is 2D
        if (mode == "declare") then
          args[idx] = p:genFullName()
        else
          args[idx] = p:genName()
        end
        idx = idx + 1
      end
    end
  end
  -- order elements by alphabetical order (good or bad?)
  --table.sort(args)
  return table.concat(args, ", ")
end
-- generate the arguments for a function definition
function BaseModel:genFuncDefArgs()
  return self:genArguments("declare")
end

-- generate the argument for a function call
function BaseModel:genFuncCallArgs()
  return self:genArguments()
end

-- generate the registers required by the model
-- NEEDS TO BE DEFINED BY THE DERIVED MODEL
-- TODO: these are usually the equilibrium macroscopic quantities,
--       they should be generated from the relaxation method
function BaseModel:genLocalVarsDef()
  return ""
end

-- return how to access/compute the velocity in a kernel
-- usually, its just by accessing velocity_name,
-- but in some models it might require additional computations,
-- so derive this function if needed
function BaseModel:genVelocity()
  return self.velocity_name
end

-- Define the SRT-BGK model
BGKModel = class(BaseModel)
function BGKModel:_init(precision, args)
  assert(args, "No arguments given to the BGK model.")
  assertTable(args)
  assert(args.size, "No size given.")
  assertTable(args.size)
  -- init the base class
  self:super("BGK", precision, "dist_func", args.size[1], args.size[2], args.size[3])
  -- relaxation time
  assert(args.tau, "No relaxation time 'tau' given.")
  self.tau = Parameter(self.precision, "tau", args.tau, "Relaxation time used in the collision")
end

-- generate the registers required by the model (here the fieqs)
function BGKModel:genLocalVarsDef()
  return model.precision .. " " .. table.concat(node:genDFNamesList(), "eq, ") .. "eq"
end

--model = BGKModel("single_precision", 32, 32, 32, 0.6)
--print(model:genFuncDefArgs(), "\n")
--print(model:genFuncCallArgs(), "\n")

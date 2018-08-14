require "merge"

-- Moment class to define a moment
Moment = class()
function Moment:_init(name, description, order, default, formula)
  -- Name of the moment
  self.name = name
  -- Description of what the moment represents
  self.description = description
  -- Order of the moment (0:scalar, 1:vector, 2:tensor...)
  self.order = order
  -- Default value of the moment
  self.default = default
  -- Formula (function) for computing the moment
  self.formula = formula
end

-- Function to generate the computations of the moment
function Moment:genComputations()
  --[[
  if(type(self.name)=='string') then
    print("generating "..self.name)
  else
    print("generating "..table.concat(self.name, ","))
  end
  --]]
  return totable(merge({model.precision, " ", self.name, " = ", self.formula()}))
end

-- Function to generate the declaration and initialisation (with default values) of the moment
function Moment:genDeclInit()
  return totable(merge({model.precision, " ", self.name, " = ", self.default}))
end

-- Definition of the density moment
DensityMoment =
  Moment(
  "rho",
  "density",
  0,
  1.0,
  function()
    local fi_list = node:genDFNamesList()
    return SUM(fi_list)
  end
)

-- Definition of the velocity moment
VelocityMoment =
  Moment(
  {"vx", "vy", "vz"},
  "velocity",
  1,
  {0.0, 0.0, 0.0},
  function()
    local fis = node:genDFNamesList()
    local eis = node.directions
    local rho = DensityMoment.name
    local sum = SUM(fis * eis)
    return {
      (1 / rho) * sum[1],
      (1 / rho) * sum[2],
      (1 / rho) * sum[3]
    }
  end
)

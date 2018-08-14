SmagorinskyDyn = DynamicsGenerator("smag", {DensityMoment, VelocityMoment})

-- generate the relaxation
function SmagorinskyDyn:genRelaxation()
  local res = {}
  -- compute the difference to equilibrium
  local fi_diffs = {}
  for i = 1, node.Q do
    local fi = node:genDistributionName(i)
    local fi_eq = node:genEquilibriumDistributionName(i)
    fi_diffs[i] = fi - fi_eq
  end
  table.insert(res, "//compute the non-equilibrium stress tensor")
  local Q = ""
  for i = 1, node.D do
    for j = i, node.D do
      local x1 = model.position_name[i]
      local x2 = model.position_name[j]
      local Pij_name = ("Pi_" .. x1 .. "_" .. x2)
      local Pij = ""
      for k = 1, node.Q do
        local e = node.directions[k]
        Pij = Pij + e[i] * e[j] * fi_diffs[k]
      end
      table.insert(res, model.precision .. " " .. Pij_name .. " = " .. Pij)
      if (i == j) then
        Q = Q + Pij_name * Pij_name
      else
        Q = Q + 2 * Pij_name * Pij_name
      end
    end
  end
  table.insert(res, model.precision .. " Q = " .. Q)

  if model.precision == "float" then
    table.insert(res, model.precision .. " S = 1.f/(6.f) * ( sqrtf( nu*nu + 18 * C*C * sqrtf(Q) ) - nu )")
  else
    table.insert(res, model.precision .. " S = 1./(6.) * ( sqrt( nu*nu + 18 * C*C * sqrt(Q) ) - nu )")
  end
  table.insert(res, "//local relaxation time")
  table.insert(res, model.precision .. " TauS = (6*( nu + S ) + 1)/2")

  local tau = "TauS"
  local X = model.position_name
  for i = 1, node.Q do
    local fi = node:genDistributionName(i)
    local fi_eq = node:genEquilibriumDistributionName(i)
    local fi_new
    if model.precision == "float" then
      fi_new = ((1 - "1.f" / tau) * fi + ("1.f" / tau) * fi_eq)
    else
      fi_new = ((1 - 1 / tau) * fi + (1 / tau) * fi_eq)
    end
    table.insert(res, memory:f_tmp(i, X) .. " = " .. fi_new)
  end
  return res
end

-- Define the parameters of the Smagorinsky model
SmagorinskyModel = class(BaseModel)
function SmagorinskyModel:_init(precision, args)
  assert(args, "No arguments given to the Smagorinsky model.")
  assertTable(args)
  assert(args.size, "No size given.")
  assertTable(args.size)
  -- init the base class
  self:super("Smag", precision, "dist_func", args.size[1], args.size[2], args.size[3])
  -- viscosity
  assert(args.nu, "No viscosity 'nu' given.")
  self.nu = Parameter(self.precision, "nu", args.nu, "Viscosity of the fluid")
  -- smagorinksy constant
  assert(args.C, "No Smagorinsky constant 'C' given")
  self.C = Parameter(self.precision, "C", args.C, "Constant used by the turbulence model")
end

-- generate the registers required by the model
function SmagorinskyModel:genLocalVarsDef()
  local fi_eqs = append(node:genDFNamesList(), "eq")
  return model.precision .. " " .. expand(fi_eqs)
end

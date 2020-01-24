-- dynamics to generate a TRT code (two relaxation time)
-- based on Irina Ginzburg model
--

-- Only density and velocity moments are required, as for BGK
dynTRT = DynamicsGenerator("TRT", {DensityMoment, VelocityMoment})

--relax one distribution function
local function TRTrelax(i, j, lambdaE, lambdaO)
  local fi = node:genDistributionName(i)
  local fj = node:genDistributionName(j)
  local fjp = fj .. "p"
  local fjm = fj .. "m"
  local fj_eq = node:genEquilibriumDistributionName(j)
  local fjeqp = fj_eq .. "p"
  local fjeqm = fj_eq .. "m"
  return memory:f_tmp(i, model.position_name) .. " = " .. (fi + lambdaE * (fjp - fjeqp) + lambdaO * (fjm - fjeqm))
end

-- The computations specific to TRT are done in the relaxation function
function dynTRT:genRelaxation()
  local res = {}
  local P = model.precision .. " "
  local allpairs = node:getPairs()
  -- compute symetric and anti-symetric distribution functions
  for i, j in pairs(allpairs.fipairs) do
    local fi = node:genDistributionName(i)
    local fj = node:genDistributionName(j)
    table.insert(res, P .. fi .. "p = " .. (fi + fj) / 2)
    table.insert(res, P .. fi .. "m = " .. (fi - fj) / 2)
  end
  -- compute symetric and anti-symetric equilibrium distribution functions
  for i, j in pairs(allpairs.fipairs) do
    local fi = node:genEquilibriumDistributionName(i)
    local fj = node:genEquilibriumDistributionName(j)
    table.insert(res, P .. fi .. "p = " .. (fi + fj) / 2)
    table.insert(res, P .. fi .. "m = " .. (fi - fj) / 2)
  end

  -- relax using TRT
  local X = model.position_name
  -- direction zero, special case
  local f0 = node:genDistributionName(allpairs.zero)
  local f0_eq = node:genEquilibriumDistributionName(allpairs.zero)
  table.insert(res, memory:f_tmp(allpairs.zero, X) .. " = " .. (f0 + model.lambdaE.name * (f0 - f0_eq)))
  --relax directions by pairs
  for i, j in pairs(allpairs.fipairs) do
    table.insert(res, TRTrelax(i, i, model.lambdaE.name, model.lambdaO.name))
    table.insert(res, TRTrelax(j, i, model.lambdaE.name, -1 * model.lambdaO.name))
  end
  --[[
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
      local Pij_name = ("Pi_"..x1.."_"..x2)
      local Pij = ""
      for k = 1, node.Q do
        local e = node.directions[k]
        Pij = Pij + e[i]*e[j]*fi_diffs[k]
      end
      table.insert(res, model.precision.." "..Pij_name.." = "..Pij)
      if (i==j) then
        Q = Q + Pij_name*Pij_name
      else
        Q = Q + 2*Pij_name*Pij_name
      end
    end
  end
  table.insert(res, model.precision.." Q = "..Q)
  
  if model.precision == 'float' then
    table.insert(res, model.precision.." S = 1.f/(6.f) * ( sqrtf( nu*nu + 18 * C*C * sqrtf(Q) ) - nu )")
  else
    table.insert(res, model.precision.." S = 1./(6.) * ( sqrt( nu*nu + 18 * C*C * sqrt(Q) ) - nu )")
  end
  table.insert(res, "//local relaxation time")
  table.insert(res, model.precision.." TauS = (6*( nu + S ) + 1)/2")

  local tau = "TauS"
  for i = 1,node.Q do
    local fi = node:genDistributionName(i)
    local fi_eq = node:genEquilibriumDistributionName(i)
    local fi_new
    if model.precision == 'float' then
      fi_new = ( (1-"1.f"/tau) * fi + ("1.f"/tau) * fi_eq )
    else
      fi_new = ( (1-1/tau) * fi + (1/tau) * fi_eq )
    end
    table.insert(res, memory:genAccessToDf_tmp_GPU(i,"idx").." = ".. fi_new )
  end
  --]]
  return res
end

-- Define the parameters of the TRT model
TRT = class(BaseModel)
function TRT:_init(precision, args)
  assert(args, "No arguments given to the TRT model.")
  assertTable(args)
  assert(args.size, "No size given.")
  assertTable(args.size)
  -- init the base class
  self:super("TRT", precision, "dist_func", args.size[1], args.size[2], args.size[3])
  -- relaxation parameters
  assert(args.lambdaE, "No relaxation parameter 'lambdaE' was given.")
  self.lambdaE = Parameter(self.precision, "lambdaE", args.lambdaE, "Even relaxation parameter")
  assert(args.lambdaO, "No relaxation parameter 'lambdaO' was given.")
  self.lambdaO = Parameter(self.precision, "lambdaO", args.lambdaO, "Odd relaxation parameter")
end

-- generate the registers required by the model
--[[
function TRT:genLocalVarsDef()
  local fips = append(node:genDFNamesList(),"p")
  local fims = append(node:genDFNamesList(),"m")
  return model.precision .. " " .. expand(fips) ..",  ".. expand(fims)
end

--]]
function TRT:genLocalVarsDef()
  local fi_eqs = append(node:genDFNamesList(), "eq")
  return model.precision .. " " .. expand(fi_eqs)
end

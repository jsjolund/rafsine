require "operators"
require "Moment"

DynamicsGenerator = class()
function DynamicsGenerator:_init(dynamics_name, required_moments)
  -- Name of the dynamic (ex: single_phase)
  self.name = dynamics_name
  -- Moments of the dfs required by this dynamic
  self.moments = required_moments
end

--generate the computation of the equilibrium distribution functions
-- TODO: depends on the model
function DynamicsGenerator:genEquilibriumDfs()
  local fieqs = {}
  --print( "Velocity : "..table.concat(model.velocity_name, ","))
  for i = 1, node.Q do
    local wi = node:getWeight(i)
    local ei = node:getDirection(i)
    local U = model.velocity_name
    local rho = model.density_name
    local formula
    if model.precision == "float" then
      formula = (wi * rho * (1 + 3 * (ei .. U) + "4.5f" * (ei .. U) ^ 2 - "1.5f" * (U .. U)))
    else
      -- incompressible formula:
      --formula = ( wi * ( rho + 1 *( 3 * (ei..U) + 4.5 * (ei..U)^2 - 1.5*(U..U) ) ) )
      formula = (wi * rho * (1 + 3 * (ei .. U) + 4.5 * (ei .. U) ^ 2 - 1.5 * (U .. U)))
    end
    table.insert(fieqs, formula)
  end
  return fieqs
end

-- generate a simple BGK relaxation
function DynamicsGenerator:genBGK()
  local res = {}
  local tau = model.tau.name
  for i = 1, node.Q do
    local fi = node:genDistributionName(i)
    local fi_eq = node:genEquilibriumDistributionName(i)
    --table.insert(res, ( (1-1/tau) * fi + (1/tau) * fi_eq ) )
    if model.precision == "float" then
      table.insert(res, ((1 - "1.f" / tau) * fi + ("1.f" / tau) * fi_eq))
    else
      table.insert(res, ((1 - 1 / tau) * fi + (1 / tau) * fi_eq))
    end
  end
  return res
end

--SinglePhaseDyn = DynamicsGenerator("single_phase", required_moments = {rho, v, ...} ??? )
SinglePhaseDyn = DynamicsGenerator("single_phase", {DensityMoment, VelocityMoment})

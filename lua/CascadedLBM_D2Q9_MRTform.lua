-- Implements the 2D cascaded LBM model as described by Daniel Lycett-Brown in his 2014 paper
-- !!!!!!!!!!!!!!!!!!!! USES THE MRT FORM NOT THE CASCADED !!!!!!!!!!!!!!!!!!!!!
--[[ BibTex citation
@article{LycettBrown2014350,
title = "Multiphase cascaded lattice Boltzmann method ",
author = "Daniel Lycett-Brown and Kai H. Luo",
journal = "Computers & Mathematics with Applications ",
volume = "67",
number = "2",
pages = "350 - 362",
year = "2014",
note = "Mesoscopic Methods for Engineering and Science (Proceedings of ICMMES-2012, Taipei, Taiwan, 23–27 July 2012) ",
doi = "http://dx.doi.org/10.1016/j.camwa.2013.08.033",
url = "http://www.sciencedirect.com/science/article/pii/S0898122113005403",
}


--]]
-- TODO: This is only for D2Q9

require "Moment"
require "ModelParameters"

-- Define the model
CLBM_D2Q9 = class(BaseModel)
function CLBM_D2Q9:_init(precision, args)
  assert(args, "No arguments given to the CLBM_D2Q9 model.")
  assertTable(args)
  assert(args.size, "No size given.")
  assertTable(args.size)
  -- init the base class
  self:super("MRT", precision, "moment", args.size[1], args.size[2])
  -- relaxation times
  assert(args.w, "No relaxation time 'w' given.")
  self.w = Parameter(self.precision, "w", args.w, "Relaxation time")
  assert(args.wb, "No relaxation time 'wb' given.")
  self.wb = Parameter(self.precision, "wb", args.wb, "Relaxation time")
  assert(args.w3, "No relaxation time 'w3' given.")
  self.w3 = Parameter(self.precision, "w3", args.w3, "Relaxation time")
  assert(args.w4, "No relaxation time 'w4' given.")
  self.w4 = Parameter(self.precision, "w4", args.w4, "Relaxation time")
  self:computeCentralMoments()
end

-- function to compute the central moments used to defined the 6 moments of the model
function CLBM_D2Q9:computeCentralMoments()
  self.moments = {{}, {}, {}}
  local fis = node:genDFNamesList()
  local eis = node.directions
  local rho = DensityMoment.name
  local Ux = genTable(9, "vx")
  local Uy = genTable(9, "vy")
  for p = 0, 2 do
    for q = 0, 2 do
      -- the moment p,q is store at the location p+1,q+1 (lua is 1-indexed)

      -- Formula for centered moments
      --self.moments[p+1][q+1] = (1/rho) * SUM( (COL(eis,1)-Ux)^p * (COL(eis,2)-Uy)^q * fis )

      -- Formula for raw moments
      self.moments[p + 1][q + 1] = (1 / rho) * SUM((COL(eis, 1)) ^ p * (COL(eis, 2)) ^ q * fis)

      --print('~M'..p..q.." = "..self.moments[p+1][q+1])
    end
  end
end

--function to access the momemts with a [0;2] indexing (instead of [1;3])
function CLBM_D2Q9:M(i, j)
  return self.moments[i + 1][j + 1]
end

-- generate a few registers for the equilibrium moments
function CLBM_D2Q9:genLocalVarsDef()
  --return model.precision.." Pxx_eq, Pxy_eq, Pyy_eq, N_eq, Jx_eq, Jy_eq"
end

-- Define the moments used by the model

TMoment =
  Moment(
  "T",
  "trace of the pressure tensor",
  0,
  0.0,
  function()
    return model:M(2, 0) + model:M(0, 2)
  end
)
NMoment =
  Moment(
  "N",
  "normal stress difference",
  0,
  0.0,
  function()
    return model:M(2, 0) - model:M(0, 2)
  end
)
PxyMoment =
  Moment(
  "Pxy",
  "off diagonal element of the stress tensor",
  0,
  0.0,
  function()
    return model:M(1, 1)
  end
)
QyxxMoment =
  Moment(
  "Qyxx",
  "fisrt third order moment",
  0,
  0.0,
  function()
    return model:M(2, 1)
  end
)
QxyyMoment =
  Moment(
  "Qxyy",
  "second third order moment",
  0,
  0.0,
  function()
    return model:M(1, 2)
  end
)
AMoment =
  Moment(
  "A",
  "fourth order moment",
  0,
  0.0,
  function()
    return model:M(2, 2)
  end
)

-- Define the dynamics of the model
dynCLBM_D2Q9 =
  DynamicsGenerator(
  "single_phase",
  {DensityMoment, VelocityMoment, TMoment, NMoment, PxyMoment, QyxxMoment, QxyyMoment, AMoment}
)

-- defines the moments that relaxes and how
function dynCLBM_D2Q9:relaxing_moments()
  local U = model.velocity_name
  local Ux = U[1]
  local Uy = U[2]
  -- moment, relaxation rate, equilibrium formula (central moment, raw moment)
  return {
    {m = PxyMoment, tau = model.w, eq = Ux * Uy},
    {m = NMoment, tau = model.w, eq = Ux ^ 2 - Uy ^ 2},
    {m = TMoment, tau = model.wb, eq = 2 / 3 + (U .. U)},
    {m = QxyyMoment, tau = model.w3, eq = Ux * (1 / 3 + Uy ^ 2)},
    {m = QyxxMoment, tau = model.w3, eq = Uy * (1 / 3 + Ux ^ 2)},
    {m = AMoment, tau = model.w4, eq = (1 / 3 + Ux ^ 2) * (1 / 3 + Uy ^ 2)}
  }
end

-- generate the relaxation of the moment
function dynCLBM_D2Q9:genRelaxationMoments()
  result = {}
  for i, moment in ipairs(self:relaxing_moments()) do
    local m = moment.m.name
    local tau = moment.tau.name
    local meq = moment.eq
    result[i] = m .. " = " .. (1 - tau) * m + tau * meq
  end
  return result
end

-- Matches a vector with a direction type
function dynCLBM_D2Q9:mask(ei, s, l)
  return math.abs(ei[1]) == s and math.abs(ei[2]) == l
end

-- generate the computation of the distribution functions from the moments
function dynCLBM_D2Q9:genDFComputations()
  local fis = {}
  for i = 1, node.Q do
    local wi = node:getWeight(i)
    local ei = node:getDirection(i)
    local U = model.velocity_name
    local Ux = U[1]
    local Uy = U[2]
    local rho = model.density_name
    local Pxy = PxyMoment.name
    local N = NMoment.name
    local T = TMoment.name
    local Qxyy = QxyyMoment.name
    local Qyxx = QyxxMoment.name
    local A = AMoment.name

    local s = ei[1] -- sigma
    local l = ei[2] -- lambda

    -- Formulas to compute distribution functions from the raw moments
    ---[[
    if self:mask(ei, 0, 0) then
      formula = rho * (1 - T + A)
    elseif self:mask(ei, 1, 0) then
      formula = rho / 2 * ((T + N) / 2 + s * Ux - s * Qxyy - A)
    elseif self:mask(ei, 0, 1) then
      formula = rho / 2 * ((T - N) / 2 + l * Uy - l * Qyxx - A)
    elseif self:mask(ei, 1, 1) then
      formula = rho / 4 * (A + s * l * Pxy + s * Qxyy + l * Qyxx)
    else
      error("unknown direction : (" .. ei[1] .. "," .. ei[2] .. ")")
    end
    --]]
    -- Formulas to compute distribution functions from the central moments
    --[[
    if     self:mask(ei, 0,0) then
      formula = rho * ( 1 - (U..U) + 4*Ux*Uy*Pxy - (Ux^2-Uy^2)/2 * N + ((U..U)-2)/2 * T + 2*Ux*Qxyy + 2*Uy*Qyxx + A )
    elseif self:mask(ei, 1,0) then
      formula = rho/2 * ( Ux^2 + s*Ux*(1-Uy^2) - (2*s*Uy + 4*Ux*Uy) * Pxy + (1+s*Ux+Ux^2-Uy^2)/2 * N 
        + (1-s*Ux-(U..U))/2 * T - (s+2*Ux) * Qxyy - 2*Uy*Qyxx - A)
    elseif self:mask(ei, 0,1) then
      formula = rho/2 * ( Uy^2 + l*Uy*(1-Ux^2) - (2*l*Ux + 4*Ux*Uy) * Pxy + (-1-l*Uy+Ux^2-Uy^2)/2 * N 
        + (1-l*Uy-(U..U))/2 * T - (l+2*Uy) * Qyxx - 2*Ux*Qxyy - A)
    elseif self:mask(ei, 1,1) then
      formula = rho/4 * ( s*l*Ux*Uy + s*Ux*Uy^2 + l*Uy*Ux^2 + ( 4*Ux*Uy+s*l+2*s*Uy+2*l*Ux ) * Pxy 
        + (-1*Ux^2+Uy^2-s*Ux+l*Uy)/2 * N + ((U..U)+s*Ux+l*Uy)/2 * T + (s+2*Ux) * Qxyy + (l+2*Uy) * Qyxx + A )
    else
      error("unknown direction : ("..ei[1]..","..ei[2]..")")
    end
    --]]
    --print("\n",formula)
    table.insert(fis, formula)
  end
  return fis
end

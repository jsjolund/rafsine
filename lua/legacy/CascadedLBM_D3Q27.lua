-- Implements the 3D cascaded LBM model as described by Daniel Lycett-Brown in his 2014 paper
-- NOTE: This is only for D3Q27
--[[ BibTex citation
@article{:/content/aip/journal/pof2/26/2/10.1063/1.4866146,
author = "Lycett-Brown, Daniel and Luo, Kai H. and Liu, Ronghou and Lv, Pengmei",
title = "Binary droplet collision simulations by a multiphase cascaded lattice Boltzmann method",
journal = "Physics of Fluids (1994-present)",
year = "2014",
volume = "26",
number = "2", 
url = "http://scitation.aip.org/content/aip/journal/pof2/26/2/10.1063/1.4866146",
doi = "http://dx.doi.org/10.1063/1.4866146" 
}
--]]
require "Moment"
require "ModelParameters"

-- Define the model
CLBM_D3Q27 = class(BaseModel)
function CLBM_D3Q27:_init(precision, args)
  assert(args, "No arguments given to the CLBM_D3Q27 model.")
  assertTable(args)
  assert(args.size, "No size given.")
  assertTable(args.size)
  -- init the base class
  self:super("MRT", precision, "moment", args.size[1], args.size[2], args.size[3])
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
function CLBM_D3Q27:computeCentralMoments()
  self.moments = {}
  local fis = node:genDFNamesList()
  local eis = node.directions
  local rho = DensityMoment.name
  local Ux = genTable(27, "vx")
  local Uy = genTable(27, "vy")
  local Uz = genTable(27, "vz")
  for p = 0, 2 do
    self.moments[p + 1] = {}
    for q = 0, 2 do
      self.moments[p + 1][q + 1] = {}
      for r = 0, 2 do
        -- the moment p,q,r is stored at the location p+1,q+1,r+1 (lua is 1-indexed)
        self.moments[p + 1][q + 1][r + 1] =
          (1 / rho) * SUM((COL(eis, 1) - Ux) ^ p * (COL(eis, 2) - Uy) ^ q * (COL(eis, 3) - Uz) ^ r * fis)
        -- raw moments
        --self.moments[p+1][q+1][r+1] = (1/rho) * SUM( (COL(eis,1))^p * (COL(eis,2))^q * (COL(eis,3))^r * fis )
      end
    end
  end
end

--function to access the momemts with a [0;2] indexing (instead of [1;3])
function CLBM_D3Q27:M(i, j, k)
  return self.moments[i + 1][j + 1][k + 1]
end

-- Define the moments used by the model
TMoment =
  Moment(
  "T",
  "trace of the pressure tensor",
  0,
  0.0,
  function()
    return model:M(2, 0, 0) + model:M(0, 2, 0) + model:M(0, 0, 2)
  end
)

NxzMoment =
  Moment(
  "Nxz",
  "normal stress difference (1 of 2)",
  0,
  0.0,
  function()
    return model:M(2, 0, 0) - model:M(0, 0, 2)
  end
)
NyzMoment =
  Moment(
  "Nyz",
  "normal stress difference (2 of 2)",
  0,
  0.0,
  function()
    return model:M(0, 2, 0) - model:M(0, 0, 2)
  end
)

PxyMoment =
  Moment(
  "Pxy",
  "off diagonal element of the stress tensor (1 of 3)",
  0,
  0.0,
  function()
    return model:M(1, 1, 0)
  end
)
PxzMoment =
  Moment(
  "Pxz",
  "off diagonal element of the stress tensor (2 of 3)",
  0,
  0.0,
  function()
    return model:M(1, 0, 1)
  end
)
PyzMoment =
  Moment(
  "Pyz",
  "off diagonal element of the stress tensor (3 of 3)",
  0,
  0.0,
  function()
    return model:M(0, 1, 1)
  end
)

QxxyMoment =
  Moment(
  "Qxxy",
  "third order moment (1 of 6)",
  0,
  0.0,
  function()
    return model:M(2, 1, 0)
  end
)
QxxzMoment =
  Moment(
  "Qxxz",
  "third order moment (2 of 6)",
  0,
  0.0,
  function()
    return model:M(2, 0, 1)
  end
)
QxyyMoment =
  Moment(
  "Qxyy",
  "third order moment (3 of 6)",
  0,
  0.0,
  function()
    return model:M(1, 2, 0)
  end
)
QyyzMoment =
  Moment(
  "Qyyz",
  "third order moment (4 of 6)",
  0,
  0.0,
  function()
    return model:M(0, 2, 1)
  end
)
QxzzMoment =
  Moment(
  "Qxzz",
  "third order moment (5 of 6)",
  0,
  0.0,
  function()
    return model:M(1, 0, 2)
  end
)
QyzzMoment =
  Moment(
  "Qyzz",
  "third order moment (6 of 6)",
  0,
  0.0,
  function()
    return model:M(0, 1, 2)
  end
)

AxxyyMoment =
  Moment(
  "Axxyy",
  "fourth order moment (1 of 3)",
  0,
  0.0,
  function()
    return model:M(2, 2, 0)
  end
)
AxxzzMoment =
  Moment(
  "Axxzz",
  "fourth order moment (2 of 3)",
  0,
  0.0,
  function()
    return model:M(2, 0, 2)
  end
)
AyyzzMoment =
  Moment(
  "Ayyzz",
  "fourth order moment (3 of 3)",
  0,
  0.0,
  function()
    return model:M(0, 2, 2)
  end
)

-- Define the dynamics of the model
dynCLBM_D3Q27 =
  DynamicsGenerator(
  "single_phase",
  {
    DensityMoment,
    VelocityMoment,
    TMoment,
    NxzMoment,
    NyzMoment,
    PxyMoment,
    PxzMoment,
    PyzMoment,
    QxxyMoment,
    QxxzMoment,
    QxyyMoment,
    QyyzMoment,
    QxzzMoment,
    QyzzMoment,
    AxxyyMoment,
    AxxzzMoment,
    AyyzzMoment
  }
)

-- defines the moments that relaxes and how
function dynCLBM_D3Q27:relaxing_moments()
  return {
    {m = TMoment, tau = model.wb, eq = 1},
    {m = NxzMoment, tau = model.w, eq = 0},
    {m = NyzMoment, tau = model.w, eq = 0},
    {m = PxyMoment, tau = model.w, eq = 0},
    {m = PxzMoment, tau = model.w, eq = 0},
    {m = PyzMoment, tau = model.w, eq = 0},
    {m = QxxyMoment, tau = model.w3, eq = 0},
    {m = QxxzMoment, tau = model.w3, eq = 0},
    {m = QxyyMoment, tau = model.w3, eq = 0},
    {m = QyyzMoment, tau = model.w3, eq = 0},
    {m = QxzzMoment, tau = model.w3, eq = 0},
    {m = QyzzMoment, tau = model.w3, eq = 0},
    {m = AxxyyMoment, tau = model.w4, eq = 1 / 9},
    {m = AxxzzMoment, tau = model.w4, eq = 1 / 9},
    {m = AyyzzMoment, tau = model.w4, eq = 1 / 9}
  }
end

-- generate the relaxation of the moments
function dynCLBM_D3Q27:genRelaxationMoments()
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
function dynCLBM_D3Q27:mask(ei, s, l, d)
  return math.abs(ei[1]) == s and math.abs(ei[2]) == l and math.abs(ei[3]) == d
end

function dynCLBM_D3Q27:genDFComputations()
  local fis = {}
  for i = 1, node.Q do
    local wi = node:getWeight(i)
    local ei = node:getDirection(i)
    local rho = model.density_name
    local U = model.velocity_name
    local Ux = U[1]
    local Uy = U[2]
    local Uz = U[3]
    local T = TMoment.name
    local Nxz = NxzMoment.name
    local Nyz = NyzMoment.name
    local Pxy = PxyMoment.name
    local Pxz = PxzMoment.name
    local Pyz = PyzMoment.name
    local Qxxy = QxxyMoment.name
    local Qxxz = QxxzMoment.name
    local Qxyy = QxyyMoment.name
    local Qyyz = QyyzMoment.name
    local Qxzz = QxzzMoment.name
    local Qyzz = QyzzMoment.name
    local Axxyy = AxxyyMoment.name
    local Axxzz = AxxzzMoment.name
    local Ayyzz = AyyzzMoment.name

    local S = ei[1] -- sigma
    local L = ei[2] -- lambda
    local D = ei[3] -- delta

    -- the following are the formulas for the "raw" moments (WIP)
    --[[
    local M222 = 0
    local M122 = 0
    local M212 = 0
    local M221 = 0
    if self:mask(ei, 0,0,0) then
      formula = rho * ( 1 - T + Axxyy + Axxzz + Ayyzz - M222 )
    elseif self:mask(ei, 1,0,0) then
      formula = rho/2 * (S*Ux + 2/3*Nxy -1/3*Nyz - 1/3*T - S*Qxyy - S*Qxzz -Axxyy -Axxzz + S*M122 + M222)
    elseif self:mask(ei, 0,1,0) then
      formula = rho/2 * (L*Uy - 1/3*Nxz 
    end
    --]]
    -- the following are the proper cascaded formulas
    ---[[
    if self:mask(ei, 0, 0, 0) then -- formula (8)
      formula =
        rho *
        (26 / 27 - (U .. U) + 4 * Ux * Uy * Pxy + 4 * Ux * Uz * Pxz + 4 * Uy * Uz * Pyz +
          1 / 3 * (-2 * Ux ^ 2 + Uy ^ 2 + Uz ^ 2) * Nxz +
          1 / 3 * (Ux ^ 2 - 2 * Uy ^ 2 + Uz ^ 2) * Nyz +
          (2 / 3 * (U .. U) - 1) * T +
          2 * Uy * (1 - Uz ^ 2) * Qxxy +
          2 * Uz * (1 - Uy ^ 2) * Qxxz +
          2 * Ux * (1 - Uz ^ 2) * Qxyy +
          2 * Uz * (1 - Ux ^ 2) * Qyyz +
          2 * Ux * (1 - Uy ^ 2) * Qxzz +
          2 * Uy * (1 - Ux ^ 2) * Qyzz +
          (1 - Uz ^ 2) * Axxyy +
          (1 - Uy ^ 2) * Axxzz +
          (1 - Ux ^ 2) * Ayyzz)
    elseif self:mask(ei, 1, 0, 0) then --formula (B1)
      formula =
        rho / 2 *
        (1 / 27 + Ux ^ 2 + S * Ux * (1 - Uy ^ 2 - Uz ^ 2) + (-4 * Ux * Uy + 2 * S * Uy * (Uz ^ 2 - 1)) * Pxy +
          (-4 * Ux * Uz + 2 * S * Uz * (Uy ^ 2 - 1)) * Pxz +
          4 * S * Ux * Uy * Uz * Pyz +
          1 / 3 * (2 * (1 + Ux ^ 2 - Uy ^ 2 - Uz ^ 2) + S * Ux * (2 - Uy ^ 2 - Uz ^ 2)) * Nxz +
          1 / 3 * (-1 - Ux ^ 2 + Uy ^ 2 + Uz ^ 2 + S * Ux * (-1 - Uy ^ 2 + 2 * Uz ^ 2)) * Nyz +
          1 / 3 * (1 - 2 * Ux ^ 2 - Uy ^ 2 - Uz ^ 2 + S * Ux * (-2 + Uy ^ 2 + Uz ^ 2)) * T +
          2 * Uy * (Uz ^ 2 - 1) * Qxxy +
          2 * Uz * (Uy ^ 2 - 1) * Qxxz +
          (S + 2 * Ux) * (Uz ^ 2 - 1) * Qxyy +
          2 * Ux * Uz * (S + Ux) * Qyyz +
          (S + 2 * Ux) * (Uy ^ 2 - 1) * Qxzz +
          2 * Ux * Uy * (S + Ux) * Qyzz +
          (Uz ^ 2 - 1) * Axxyy +
          (Uy ^ 2 - 1) * Axxzz +
          (Ux ^ 2 + S * Ux) * Ayyzz)
    elseif self:mask(ei, 0, 1, 0) then -- formula (B2)
      formula =
        rho / 2 *
        (1 / 27 + Uy ^ 2 + L * Uy * (1 - Ux ^ 2 - Uz ^ 2) + (-4 * Ux * Uy + 2 * L * Ux * (Uz ^ 2 - 1)) * Pxy +
          4 * L * Ux * Uy * Uz * Pxz +
          (-4 * Uy * Uz + 2 * L * Uz * (Ux ^ 2 - 1)) * Pyz +
          1 / 3 * (-1 + Ux ^ 2 - Uy ^ 2 + Uz ^ 2 + L * Uy * (-1 - Ux ^ 2 + 2 * Uz ^ 2)) * Nxz +
          1 / 3 * (2 * (1 - Ux ^ 2 + Uy ^ 2 - Uz ^ 2) + L * Uy * (2 - Ux ^ 2 - Uz ^ 2)) * Nyz +
          1 / 3 * (1 - Ux ^ 2 - 2 * Uy ^ 2 - Uz ^ 2 + L * Uy * (-2 + Ux ^ 2 + Uz ^ 2)) * T +
          (L + 2 * Uy) * (Uz ^ 2 - 1) * Qxxy +
          2 * Uy * Uz * (L + Uy) * Qxxz +
          2 * Ux * (Uz ^ 2 - 1) * Qxyy +
          2 * Uz * (Ux ^ 2 - 1) * Qyyz +
          2 * Ux * Uy * (L + Uy) * Qxzz +
          (L + 2 * Uy) * (Ux ^ 2 - 1) * Qyzz +
          (Uz ^ 2 - 1) * Axxyy +
          (Uy ^ 2 + L * Uy) * Axxzz +
          (Ux ^ 2 - 1) * Ayyzz)
    elseif self:mask(ei, 0, 0, 1) then -- formula (B3)
      formula =
        rho / 2 *
        (1 / 27 + Uz ^ 2 + D * Uz * (1 - Ux ^ 2 - Uy ^ 2) + 4 * D * Ux * Uy * Uz * Pxy +
          (-4 * Ux * Uz + 2 * D * Ux * (Uy ^ 2 - 1)) * Pxz +
          (-4 * Uy * Uz + 2 * D * Uy * (Ux ^ 2 - 1)) * Pyz +
          1 / 3 * (-1 + Ux ^ 2 + Uy ^ 2 - Uz ^ 2 + D * Uz * (-1 - Ux ^ 2 + 2 * Uy ^ 2)) * Nxz +
          1 / 3 * (-1 + Ux ^ 2 + Uy ^ 2 - Uz ^ 2 + D * Uz * (-1 + 2 * Ux ^ 2 - Uy ^ 2)) * Nyz +
          1 / 3 * (1 - Ux ^ 2 - Uy ^ 2 - 2 * Uz ^ 2 + D * Uz * (-2 + Ux ^ 2 + Uy ^ 2)) * T +
          2 * Uy * Uz * (D + Uz) * Qxxy +
          (D + 2 * Uz) * (Uy ^ 2 - 1) * Qxxz +
          2 * Ux * Uz * (D + Uz) * Qxyy +
          (D + 2 * Uz) * (Ux ^ 2 - 1) * Qyyz +
          2 * Ux * (Uy ^ 2 - 1) * Qxzz +
          2 * Uy * (Ux ^ 2 - 1) * Qyzz +
          (Uz ^ 2 + D * Uz) * Axxyy +
          (Uy ^ 2 - 1) * Axxzz +
          (Ux ^ 2 - 1) * Ayyzz)
    elseif self:mask(ei, 1, 1, 0) then -- formula (B4)
      formula =
        rho / 4 *
        (-1 / 27 + S * L * Ux * Uy + L * Ux ^ 2 * Uy + S * Ux * Uy ^ 2 +
          (4 * Ux * Uy + 2 * (L * Ux + S * Uy) * (1 - Uz ^ 2) - S * L * Uz ^ 2) * Pxy -
          Uy * Uz * (2 * S * L + 4 * L * Ux + 2 * S * Uy) * Pxz -
          Ux * Uz * (2 * S * L + 2 * L * Ux + 4 * S * Uy) * Pyz +
          1 / 3 *
            (-S * Ux + 2 * L * Uy - Ux ^ 2 + 2 * Uy ^ 2 + S * L * Ux * Uy + L * Ux ^ 2 * Uy + S * Ux * Uy ^ 2 +
              S * Ux * Uz ^ 2 -
              2 * L * Uy * Uz ^ 2) *
            Nxz +
          1 / 3 *
            (2 * S * Ux - L * Uy + 2 * Ux ^ 2 - Uy ^ 2 + S * L * Ux * Uy + L * Ux ^ 2 * Uy + S * Ux * Uy ^ 2 -
              2 * S * Ux * Uz ^ 2 +
              L * Uy * Uz ^ 2) *
            Nyz +
          1 / 3 *
            (S * Ux + L * Uy + Ux ^ 2 + Uy ^ 2 - S * L * Ux * Uy - L * Ux ^ 2 * Uy - S * Ux * Uy ^ 2 - S * Ux * Uz ^ 2 -
              L * Uy * Uz ^ 2) *
            T +
          (L + 2 * Uy) * (1 - Uz ^ 2) * Qxxy -
          2 * Uy * Uz * (L + Uy) * Qxxz +
          (S + 2 * Ux) * (1 - Uz ^ 2) * Qxyy -
          2 * Ux * Uz * (S + Ux) * Qyyz -
          (S + 2 * Ux) * (L * Uy + Uy ^ 2) * Qxzz -
          (L + 2 * Uy) * (S * Ux + Ux ^ 2) * Qyzz +
          (1 - Uz ^ 2) * Axxyy -
          (L * Uy + Uy ^ 2) * Axxzz -
          (S * Ux + Ux ^ 2) * Ayyzz)
    elseif self:mask(ei, 1, 0, 1) then -- formula (B5)
      formula =
        rho / 4 *
        (-1 / 27 + S * D * Ux * Uz + D * Ux ^ 2 * Uz + S * Ux * Uz ^ 2 +
          (2 * D * Ux + 2 * S * Uz - S * D * Uy ^ 2 + 4 * Ux * Uz - 2 * D * Ux * Uy ^ 2 - 2 * S * Uy ^ 2 * Uz) * Pxz -
          Uy * Uz * (2 * S * D + 4 * D * Ux + 2 * S * Uz) * Pxy -
          Ux * Uy * (2 * S * D + 2 * D * Ux + 4 * S * Uz) * Pyz +
          1 / 3 *
            (-S * Ux + 2 * D * Uz - Ux ^ 2 + 2 * Uz ^ 2 + S * D * Ux * Uz + D * Ux ^ 2 * Uz + S * Ux * Uy ^ 2 +
              S * Ux * Uz ^ 2 -
              2 * D * Uy ^ 2 * Uz) *
            Nxz +
          1 / 3 *
            (-S * Ux - D * Uz - Ux ^ 2 - Uy ^ 2 - 2 * S * D * Ux * Uz - 2 * D * Ux ^ 2 * Uz + S * Ux * Uy ^ 2 -
              2 * S * Ux * Uz ^ 2 +
              D * Uy ^ 2 * Uz) *
            Nyz +
          1 / 3 *
            (S * Ux + D * Uz + Ux ^ 2 + Uz ^ 2 - S * D * Ux * Uz - D * Ux ^ 2 * Uz - S * Ux * Uy ^ 2 - S * Ux * Uz ^ 2 -
              D * Uy ^ 2 * Uz) *
            T -
          2 * Uy * Uz * (D + Uz) * Qxxy -
          (D + 2 * Uz) * (1 - Uy ^ 2) * Qxxz -
          (S + 2 * Ux) * (D * Uz + Uz ^ 2) * Qxyy -
          (D + 2 * Uz) * (S * Ux + Ux ^ 2) * Qyyz +
          (S + 2 * Ux) * (1 - Uy ^ 2) * Qxzz -
          2 * Ux * Uy * (S + Ux) * Qyzz -
          (D * Uz + Uz ^ 2) * Axxyy +
          (1 - Uy ^ 2) * Axxzz -
          (S * Ux + Ux ^ 2) * Ayyzz)
    elseif self:mask(ei, 0, 1, 1) then -- formula (B6)
      formula =
        rho / 4 *
        (-1 / 27 + L * D * Uy * Uz + D * Uy ^ 2 * Uz + L * Uy * Uz ^ 2 +
          (2 * D * Uy + 2 * L * Uz - L * D * Ux ^ 2 + 4 * Uy * Uz - 2 * D * Ux ^ 2 * Uy - 2 * L * Ux ^ 2 * Uz) * Pyz -
          Ux * Uz * (2 * L * D + 4 * D * Uy + 2 * L * Uz) * Pxy -
          Ux * Uy * (2 * L * D + 2 * D * Uy + 4 * L * Uz) * Pxz +
          1 / 3 *
            (-L * Uy - D * Uz - Uy ^ 2 - Uz ^ 2 - 2 * L * D * Uy * Uz + L * Ux ^ 2 * Uy + D * Ux ^ 2 * Uz -
              2 * D * Uy ^ 2 * Uz -
              2 * L * Uy * Uz ^ 2) *
            Nxz +
          1 / 3 *
            (-L * Uy + 2 * D * Uz - Uy ^ 2 + 2 * Uz ^ 2 + L * D * Uy * Uz + L * Ux ^ 2 * Uy - 2 * D * Ux ^ 2 * Uz +
              D * Uy ^ 2 * Uz +
              L * Uy * Uz ^ 2) *
            Nyz +
          1 / 3 *
            (L * Uy + D * Uz + Uy ^ 2 + Uz ^ 2 - L * D * Uy * Uz - L * Ux ^ 2 * Uy - D * Ux ^ 2 * Uz - D * Uy ^ 2 * Uz -
              L * D * Uy * Uz ^ 2) *
            T -
          (L + 2 * Uy) * (D * Uz + Uz ^ 2) * Qxxy -
          (D + 2 * Uz) * (L * Uy + Uy ^ 2) * Qxxz -
          2 * Ux * Uz * (D + Uz) * Qxyy +
          (D + 2 * Uz) * (1 - Ux ^ 2) * Qyyz -
          2 * Ux * Uy * (L + Uy) * Qxzz +
          (D + 2 * Uy) * (1 - Ux ^ 2) * Qyzz -
          (D * Uz + Uz ^ 2) * Axxyy -
          (L * Uy + Uy ^ 2) * Axxzz +
          (1 - Ux ^ 2) * Ayyzz)
    elseif self:mask(ei, 1, 1, 1) then -- formula (B7)
      formula =
        rho / 8 *
        (1 / 27 + S * L * D * Ux * Uy * Uz +
          (S * L * Uz ^ 2 + 2 * L * D * Ux * Uz + 2 * S * D * Uy * Uz + 2 * L * Ux * Uz ^ 2 + 2 * S * Uy * Uz ^ 2 +
            4 * D * Ux * Uy * Uz) *
            Pxy +
          (S * L * Uy ^ 2 + 2 * L * D * Ux * Uy + 2 * S * L * Uy * Uz + 2 * D * Ux * Uy ^ 2 + 2 * S * Uy ^ 2 * Uz +
            4 * L * Ux * Uy * Uz) *
            Pxz +
          (L * D * Ux ^ 2 + 2 * S * D * Ux * Uy + 2 * S * L * Ux * Uz + 2 * D * Ux ^ 2 * Uy + 2 * L * Ux ^ 2 * Uz +
            4 * S * Ux * Uy * Uz) *
            Pyz +
          1 / 3 *
            (-S * L * Ux * Uy - S * D * Ux * Uz + 2 * L * D * Uy * Uz - L * Ux ^ 2 * Uy - D * Ux ^ 2 * Uz -
              S * Ux * Uy ^ 2 +
              2 * D * Uy ^ 2 * Uz -
              S * Ux * Uz ^ 2 +
              2 * L * Uy * Uz ^ 2) *
            Nxz +
          1 / 3 *
            (-S * L * Ux * Uy + 2 * S * D * Ux * Uz - L * D * Uy * Uz - L * Ux ^ 2 * Uy + 2 * D * Ux ^ 2 * Uz -
              S * Ux * Uy ^ 2 -
              D * Uy ^ 2 * Uz +
              2 * S * Ux * Uz ^ 2 -
              L * Uy * Uz ^ 2) *
            Nyz +
          1 / 3 *
            (S * L * Ux * Uy + S * D * Ux * Uz + L * D * Uy * Uz + L * Ux ^ 2 * Uy + D * Ux ^ 2 * Uz + S * Ux * Uy ^ 2 +
              D * Uy ^ 2 * Uz +
              S * Ux * Uz ^ 2 +
              L * Uy * Uz ^ 2) *
            T +
          (L + 2 * Uy) * (D * Uz + Uz ^ 2) * Qxxy +
          (D + 2 * Uz) * (L * Uy + Uy ^ 2) * Qxxz +
          (S + 2 * Ux) * (D * Uz + Uz ^ 2) * Qxyy +
          (D + 2 * Uz) * (S * Ux + Ux ^ 2) * Qyyz +
          (S + 2 * Ux) * (L * Uy + Uy ^ 2) * Qxzz +
          (L + 2 * Uy) * (S * Ux + Ux ^ 2) * Qyzz +
          (D * Uz + Uz ^ 2) * Axxyy +
          (L * Uy + Uy ^ 2) * Axxzz +
          (S * Ux + Ux ^ 2) * Ayyzz)
    else
      error("unknown direction : (" .. ei[1] .. "," .. ei[2] .. "," .. ei[3] .. ")")
    end
    --]]

    --print("\n",formula)
    table.insert(fis, formula)
  end
  return fis
end

-- Implements the MRT model for D2Q9 implemented in the paper from Dellar 2002
-- This implementation does not require collision matrix explicitely
--[[ BibTex citation
@article{Dellar2003351,
title = "Incompressible limits of lattice Boltzmann equations using multiple relaxation times ",
journal = "Journal of Computational Physics ",
volume = "190",
number = "2",
pages = "351 - 370",
year = "2003",
note = "",
issn = "0021-9991",
doi = "http://dx.doi.org/10.1016/S0021-9991(03)00279-1",
url = "http://www.sciencedirect.com/science/article/pii/S0021999103002791",
author = "Paul J. Dellar"
}
--]]
require "Moment"
require "ModelParameters"

-- Define the MRT model
-- TODO: This is only for D2Q9
MRTDellar2002 = class(BaseModel)
function MRTDellar2002:_init(precision, args)
  assert(args, "No arguments given to the MRTDellar2002 model.")
  assertTable(args)
  assert(args.size, "No size given.")
  assertTable(args.size)
  -- init the base class
  self:super("MRT", precision, "moment", args.size[1], args.size[2])
  -- relaxation times
  assert(args.tau, "No relaxation time 'tau' given.")
  self.tau = Parameter(self.precision, "tau", args.tau, "Relaxation time of the momentum flux")
  assert(args.tauJ, "No relaxation time 'tauJ' given.")
  self.tauJ =
    Parameter(self.precision, "tauJ", args.tauJ, "Relaxation time of the non-hydrodynamic vector J (Dellar 2002)")
  assert(args.tauN, "No relaxation time 'tauN' given.")
  self.tauN =
    Parameter(self.precision, "tauN", args.tauN, "Relaxation time of the non-hydrodynamic scalar N (Dellar 2002)")
  -- weight used in the computation of the non-hydrodynamics moments (Dellar 2002)
  self.gis = {1, -2, -2, -2, -2, 4, 4, 4, 4}
end

-- generate a few registers for the equilibrium moments
function MRTDellar2002:genLocalVarsDef()
  return model.precision .. " Pxx_eq, Pxy_eq, Pyy_eq, N_eq, Jx_eq, Jy_eq"
end

-- Definition of the moments used in the paper

-- Definition of the momentum flux (stress-tensor)
MomentumFlux =
  Moment(
  {"Pxx", "Pxy", "Pyy"},
  "momentum flux",
  2,
  {0.0, 0.0, 0.0},
  function()
    local fis = node:genDFNamesList()
    local eis = node.directions
    local Pxx = 0
    local Pxy = 0
    local Pyy = 0
    for i, fi in ipairs(fis) do
      Pxx = Pxx + eis[i][1] * eis[i][1] * fi
      Pxy = Pxy + eis[i][1] * eis[i][2] * fi
      Pyy = Pyy + eis[i][2] * eis[i][2] * fi
    end
    return {Pxx, Pxy, Pyy}
  end
)

-- Definition of the non-hydrodynamic scalar moment N (Dellar 2002)
NMoment =
  Moment(
  "N",
  "non-hydrodynamic scalar (Dellar 2002)",
  0,
  0.0,
  function()
    local fis = node:genDFNamesList()
    local N = 0
    for i, fi in ipairs(fis) do
      N = N + model.gis[i] * fi
    end
    return N
  end
)

-- Definition of the non-hydrodynamic vecotor moment J (Dellar 2002)
JMoment =
  Moment(
  {"Jx", "Jy"},
  "non-hydrodynamic vector (Dellar 2002)",
  1,
  {0.0, 0.0},
  function()
    local fis = node:genDFNamesList()
    local eis = node.directions
    local Jx = 0
    local Jy = 0
    for i, fi in ipairs(fis) do
      Jx = Jx + model.gis[i] * eis[i][1] * fi
      Jy = Jy + model.gis[i] * eis[i][2] * fi
    end
    return {Jx, Jy}
  end
)

-- Define the dynamics of the model
dynDellar2002 = DynamicsGenerator("single_phase", {DensityMoment, VelocityMoment, MomentumFlux, NMoment, JMoment})

--generate the computation of the equilibrium moments
function dynDellar2002:genEquilibriumMoments()
  local third = "1.f/3.f"
  if model.precision == "double" then
    third = "1.0/3.0"
  end
  local rho = model.density_name
  local V = model.velocity_name
  return {
    "Pxx_eq = " .. (third * rho + rho * V[1] * V[1]),
    "Pxy_eq = " .. (rho * V[1] * V[2]),
    "Pyy_eq = " .. (third * rho + rho * V[2] * V[2]),
    "N_eq = 0",
    "Jx_eq = 0",
    "Jy_eq = 0"
  }
end

-- generate the relaxation of the moment (both hydrodynamics and non-hydrodynamics)
function dynDellar2002:genRelaxationMoments()
  return {
    "Pxx = Pxx - (Pxx-Pxx_eq)/(" .. model.tau.name .. ")",
    "Pxy = Pxy - (Pxy-Pxy_eq)/(" .. model.tau.name .. ")",
    "Pyy = Pyy - (Pyy-Pyy_eq)/(" .. model.tau.name .. ")",
    "N = N - (N-N_eq)/(" .. model.tauN.name .. ")",
    "Jx = Jx - (Jx-Jx_eq)/(" .. model.tauJ.name .. ")",
    "Jy = Jy - (Jy-Jy_eq)/(" .. model.tauJ.name .. ")"
  }
end

-- generate the computation of the distribution functions from the moments
function dynDellar2002:genDFComputations()
  local fis = {}
  for i = 1, node.Q do
    local wi = node:getWeight(i)
    local ei = node:getDirection(i)
    local U = model.velocity_name
    local rho = model.density_name
    local Pxx = "Pxx"
    local Pxy = "Pxy"
    local Pyy = "Pyy"
    local N = "N"
    local J = {"Jx", "Jy"}
    local gi = model.gis[i]
    local formula
    if model.precision == "float" then
      formula =
        wi *
        (rho * (2 - "3.f/2.f" * (ei .. ei)) + 3 * rho * (ei .. U) +
          "9.f/2.f" * (Pxx * ei[1] ^ 2 + 2 * ei[1] * ei[2] * Pxy + Pyy * ei[2] ^ 2) -
          "3.f/2.f" * (Pxx + Pyy) +
          "1.f/4.f" * gi * N +
          "3.f/8.f" * gi * (ei .. J))
    else
      formula =
        wi *
        (rho * (2 - 3 / 2 * (ei .. ei)) + 3 * rho * (ei .. U) +
          9 / 2 * (Pxx * ei[1] ^ 2 + 2 * ei[1] * ei[2] * Pxy + Pyy * ei[2] ^ 2) -
          3 / 2 * (Pxx + Pyy) +
          1 / 4 * gi * N +
          3 / 8 * gi * (ei .. J))
    end
    table.insert(fis, formula)
  end
  return fis
end

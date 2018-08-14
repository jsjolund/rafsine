-- Implements the MRT model for D2Q9 implemented in the paper from Lallemand 2000
-- This implementation uses a collision matrix
--[[ BibTex citation
@article{PhysRevE.61.6546,
  title = {Theory of the lattice Boltzmann method: Dispersion, dissipation, isotropy, Galilean invariance, and stability},
  author = {Lallemand, Pierre and Luo, Li-Shi},
  journal = {Phys. Rev. E},
  volume = {61},
  issue = {6},
  pages = {6546--6562},
  year = {2000},
  month = {Jun},
  doi = {10.1103/PhysRevE.61.6546},
  url = {http://link.aps.org/doi/10.1103/PhysRevE.61.6546},
  publisher = {American Physical Society}
}
--]]
require "Moment"
require "ModelParameters"
local matrix = require "matrix"

-- Define the model
-- TODO: This is only for D2Q9
MRTLallemand2000 = class(BaseModel)
function MRTLallemand2000:_init(precision, args)
  assert(args, "No arguments given to the MRTLallemand2000 model.")
  assertTable(args)
  assert(args.size, "No size given.")
  assertTable(args.size)
  -- init the base class
  self:super("MRT", precision, "moment", args.size[1], args.size[2])
  -- relaxation times
  assert(args.s2, "No relaxation parameter 's2' given.")
  self.s2 = Parameter(self.precision, "s2", args.s2, "Relaxation parameter linked to the bulk viscosity")
  assert(args.s8, "No relaxation parameter 's8' given.")
  self.s8 = Parameter(self.precision, "s8", args.s8, "Relaxation parameter linked to the shear viscosity")
  assert(args.s3, "No relaxation parameter 's3' given.")
  self.s3 = Parameter(self.precision, "s3", args.s3, "Non-hydrodynamic relaxation parameter (see paper)")
  assert(args.s5, "No relaxation parameter 's5' given.")
  self.s5 = Parameter(self.precision, "s5", args.s5, "Non-hydrodynamic relaxation parameter (see paper)")
  assert(args.s7, "No relaxation parameter 's7' given.")
  self.s7 = Parameter(self.precision, "s7", args.s7, "Non-hydrodynamic relaxation parameter (see paper)")
  --[[
  assert(args.alpha3, "No parameter 'alpha3' given.")
  self.alpha3 = Parameter(self.precision, "alpha3", args.alpha3, "adjustable parameter (see paper)")
  assert(args.gamma4, "No parameter 'gamma4' given.")
  self.gamma4 = Parameter(self.precision, "gamma4", args.gamma4, "adjustable parameter (see paper)")
  --]]
  -- parameters used by the model
  self.c1 = -2
  self.alpha2 = -8
  self.gamma1 = 2 / 3
  self.gamma2 = 18
  self.gamma3 = 2 / 3
  -- states that the moments should be computed using a matrix
  self.matrix_transformation = true
  -- values of the transformation matrix
  self.m = {}
  -- name of the moments computed by the transformation matrix (each row)
  self.moments_names = matrix {{self.density_name, "E", "E2", "Jx", "Qx", "Jy", "Qy", "Pxx", "Pxy"}} ^ "T"
  self:initTransformationMatrix()
  -- redefine the velocity
  --self.v = {"Jx/rho","Jy/rho"}
  self.v = vector({"Jx" / self.density_name, "Jy" / self.density_name})
end

-- redefine the velocity from the momentum {Jx,Jy}
function MRTLallemand2000:genVelocity()
  return self.v
end

function MRTLallemand2000:initTransformationMatrix()
  -- Define each moment
  --density moment
  local ket_rho = vector()
  -- energy moment
  local ket_e = vector()
  -- energy square (related) moment
  local ket_e2 = vector()
  -- x-component of momentum
  local ket_j_x = vector()
  -- x-component of energy flux
  local ket_q_x = vector()
  -- y-component of momentum
  local ket_j_y = vector()
  -- y-component of energy flux
  local ket_q_y = vector()
  -- diagonal stress-tensor
  local ket_p_x_x = vector()
  -- off-diagonal stress-tensor
  local ket_p_x_y = vector()

  -- Compute each moment
  print("Generating all moments")
  for i, ei in pairs(node.directions) do
    ket_rho[i] = norm(ei, 0)
    ket_e[i] = -4 * norm(ei, 0) + 3 * (ei .. ei)
    ket_e2[i] = 4 * norm(ei, 0) - 21 / 2 * (ei .. ei) + 9 / 2 * (ei .. ei) ^ 2
    ket_j_x[i] = ei[1]
    ket_q_x[i] = (-5 * norm(ei, 0) + 3 * (ei .. ei)) * ei[1]
    ket_j_y[i] = ei[2]
    ket_q_y[i] = (-5 * norm(ei, 0) + 3 * (ei .. ei)) * ei[2]
    ket_p_x_x[i] = ei[1] ^ 2 - ei[2] ^ 2
    ket_p_x_y[i] = ei[1] * ei[2]
  end
  -- Regroup all the moments in a table
  self.M =
    matrix {
    ket_rho,
    ket_e,
    ket_e2,
    ket_j_x,
    ket_q_x,
    ket_j_y,
    ket_q_y,
    ket_p_x_x,
    ket_p_x_y
  }

  -- Print each moment (components)
  print("Transformation matrix:")
  --print(self.M)
  printMatrix(self.M)
  for i = 1, node.Q do
    print("  |" .. self.moments_names[i][1] .. ">", "=", expand(self.M[i]))
  end

  -- Check that they are all orthogonal
  ---[[
  print("Checking orthogonality of the moments...")
  local all_good = true
  for i = 1, node.Q do
    for j = 1, node.Q do
      if i ~= j then
        --print(self.moments_names[i][1].." . "..self.moments_names[j][1])
        --compute the scalar product of the two moments
        local sp = scalar_product(self.M[i], self.M[j])
        if sp ~= 0 then
          all_good = false
          print(
            "The two moments " ..
              self.moments_names[i][1] .. " and " .. self.moments_names[j][1] .. " are not orthogonal"
          )
          print(self.moments_names[i][1] .. " = " .. expand(self.M[i]))
          print(self.moments_names[j][1] .. " = " .. expand(self.M[j]))
          print(self.moments_names[i][1] .. " . " .. self.moments_names[j][1] .. " = " .. sp)
        end
      end
    end
  end
  if all_good then
    print("All the moments are mutually orthogonal to each others")
  else
    print("Some of the moments are not orthogonal, check their components")
  end
  --]]

  print("Inverting the transformation matrix")
  self.invM = matrix.invert(self.M)
  -- removing round up errors
  local cutoff = 1e-10
  for i = 1, node.Q do
    for j = 1, node.Q do
      if math.abs(self.invM[i][j]) < cutoff then
        self.invM[i][j] = 0
      end
    end
  end
  print("Invert : ")
  --print(self.invM)
  printMatrix(self.invM)

  print("Multiplying transformation matrix by its inverse:")
  test = self.M * self.invM
  -- removing round up errors
  local cutoff = 1e-10
  for i = 1, node.Q do
    for j = 1, node.Q do
      if math.abs(test[i][j]) < cutoff then
        test[i][j] = 0
      end
    end
  end
  print(test)

  -- Transforming to rational expression
  -- TODO: do this during matrix inversion computation
  ---[[
  for i = 1, node.Q do
    for j = 1, node.Q do
      if self.invM[i][j] ~= math.floor(self.invM[i][j]) then
        local denominator = round(1 / self.invM[i][j])
        if (1 / denominator - self.invM[i][j] < 0.0001) then
          local dot
          if self.precision == "float" then
            dot = ".f"
          else
            dot = "."
          end
          local numerator = "1" .. dot
          if denominator < 0 then
            numerator = "-" .. numerator
            denominator = -denominator
          end
          self.invM[i][j] = numerator .. "/" .. denominator .. dot
        end
      end
    end
  end
  --]]
  print("Inverted Matrix, rational expression")
  --print(self.invM)
  printMatrix(self.invM)
end

-- generate a few registers for the equilibrium moments
function MRTLallemand2000:genLocalVarsDef()
  return self.precision .. " E_eq, E2_eq, Qx_eq, Qy_eq, Pxx_eq, Pxy_eq"
end

-- Define the dynamics of the model
dynLallemand2000 = DynamicsGenerator("single_phase", {DensityMoment, VelocityMoment})

--generate the computation of the equilibrium moments
function dynLallemand2000:genEquilibriumMoments()
  local rho = model.density_name
  local J = vector({"Jx", "Jy"})
  local V = vector({"Jx/rho", "Jy/rho"})
  return {
    --[[
    -- The following equations (see paper) describes an imcompressible LBM model
    "E_eq = "..( 1/4 * model.alpha2 * rho + 1/6 * model.gamma2 * (J..J) ) ,
    "E2_eq = "..( 1/4 * model.alpha3.name * rho + 1/6 * model.gamma4.name * (J..J) ) ,
    "Qx_eq = "..( 1/2 * model.c1 * J[1] ) ,
    "Qy_eq = "..( 1/2 * model.c1 * J[2] ) ,
    "Pxx_eq = "..( 1/2 * model.gamma1 * (J[1]^2 - J[2]^2 ) ) ,
    "Pxy_eq = "..( 1/2 * model.gamma3 * (J[1]*J[2]) )
    --]]
    --The following equations recover the usual slightly compressible LBM
    "E_eq = " .. (rho * (3 * (V .. V) - 2)),
    "E2_eq = " .. (rho * (1 - 3 * (V .. V))),
    "Qx_eq = " .. (-1 * J[1]),
    "Qy_eq = " .. (-1 * J[2]),
    "Pxx_eq = " .. (rho * (V[1] ^ 2 - V[2] ^ 2)),
    "Pxy_eq = " .. (rho * (V[1] * V[2]))
  }
end

--helper function to relax moments
function dynLallemand2000:relax(name, s)
  return name .. " = " .. (name - s * (name - (name .. "_eq")))
end

-- generate the relaxation of the moments
function dynLallemand2000:genRelaxationMoments()
  return {
    self:relax("E", model.s2.name),
    self:relax("E2", model.s3.name),
    self:relax("Qx", model.s5.name),
    self:relax("Qy", model.s7.name),
    self:relax("Pxx", model.s8.name),
    self:relax("Pxy", model.s8.name)
  }
end

--[[
function dynLallemand2000:genInitMoments()
  return {
    "Jx = rho*vx",
    "Jy = rho*vy",
    "E  = -2*rho+3*(Jx*Jx+Jy*Jy)",
    "E2 =  1*rho-3*(Jx*Jx+Jy*Jy)",
    "Qx = -Jx",
    "Qy = -Jy",
    --"Pxx = (Jx*Jx-Jy*Jy)-2*rho/(3*s8)*(DxVx-DyVy)",
    --"Pxy = Jx*Jy - 1*rho/(3*s8)*(DyVx+DxVy)"
    "Pxx = (Jx*Jx-Jy*Jy)",
    "Pxy = Jx*Jy"
  }
end
--]]

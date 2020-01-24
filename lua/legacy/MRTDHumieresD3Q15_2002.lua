-- Implements the MRT model for D3Q15 implemented in the paper from D'Humieres et. al. 2002
-- This implementation uses a collision matrix
--[[ BibTex citation
@ARTICLE{dhumieres:02,
    author   = {d'{Humi\`eres}, Dominique and Ginzburg, Irina and Krafczyk, Manfred and Lallemand, Pierre and Luo, Li-Shi},
    title    = {Multiple-Relaxation-Time Lattice {B}oltzmann Models in Three Dimensions},
    journal  = {Phil. Trans. R. Soc. A},
    year     = {2002},
    volume   = {360},
    pages    = {437--451},
}
--]]
require "pl"
utils.import "pl.class"
require "helpers"
require "Moment"
require "ModelParameters"
local matrix = require "matrix"

-- Define the model
-- TODO: This is only for D3Q15
MRTDHumieresD3Q15_2002 = class(BaseModel)
function MRTDHumieresD3Q15_2002:_init(precision, args)
  assert(args, "No arguments given to the MRTDHumieresD3Q15_2002 model.")
  assertTable(args)
  assert(args.size, "No size given.")
  assertTable(args.size)
  -- init the base class
  self:super("MRT", precision, "moment", args.size[1], args.size[2], args.size[3])
  -- relaxation times
  assert(args.s1, "No relaxation parameter 's1' given.")
  self.s1 = Parameter(self.precision, "s1", args.s1, "Relaxation parameter linked to the bulk viscosity")
  assert(args.s9, "No relaxation parameter 's9' given.")
  self.s9 = Parameter(self.precision, "s9", args.s9, "Relaxation parameter linked to the shear viscosity")
  self.s11 = Parameter(self.precision, "s11", args.s9, "relaxation parameter equal to s9 (see paper)")
  assert(args.s2, "No relaxation parameter 's2' given.")
  self.s2 = Parameter(self.precision, "s2", args.s2, "Non-hydrodynamic relaxation parameter (see paper)")
  assert(args.s4, "No relaxation parameter 's4' given.")
  self.s4 = Parameter(self.precision, "s4", args.s4, "Non-hydrodynamic relaxation parameter (see paper)")
  assert(args.s14, "No relaxation parameter 's14' given.")
  self.s14 = Parameter(self.precision, "s14", args.s14, "Non-hydrodynamic relaxation parameter (see paper)")
  assert(args.We, "No parameter 'We' given.")
  self.We = args.We
  assert(args.Wej, "No parameter 'Wej' given.")
  self.Wej = args.Wej
  -- parameters used by the model
  -- average density of the fluid (that used in the initialisation)
  self.rho0 = 1.0
  -- states that the moments should be computed using a matrix
  self.matrix_transformation = true
  -- m will store the values of the transformation matrix
  self.m = {}
  -- name of the moments computed by the transformation matrix (each row)
  self.moments_names =
    matrix {
    {self.density_name, "E", "E2", "Jx", "Qx", "Jy", "Qy", "Jz", "Qz", "Pxx", "Pww", "Pxy", "Pyz", "Pxz", "Mxyz"}
  } ^ "T"
  self:initTransformationMatrix()
  -- redefine the velocity
  self.v = vector({"Jx" / self.density_name, "Jy" / self.density_name, "Jz" / self.density_name})
end

-- redefine the velocity from the momentum {Jx,Jy}
function MRTDHumieresD3Q15_2002:genVelocity()
  return self.v
end

function MRTDHumieresD3Q15_2002:initTransformationMatrix()
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
  -- z-component of momentum
  local ket_j_z = vector()
  -- z-component of energy flux
  local ket_q_z = vector()
  -- (3*) diagonal stress-tensor
  local ket_p_x_x = vector()
  -- stress-tensor Pww = Pyy - Pzz
  local ket_p_w_w = vector()
  -- off-diagonal stress-tensor
  local ket_p_x_y = vector()
  -- off-diagonal stress-tensor
  local ket_p_y_z = vector()
  -- off-diagonal stress-tensor
  local ket_p_z_x = vector()
  -- third order moment
  local ket_m_x_y_z = vector()

  -- Compute each moment
  print("Generating all moments")
  for i, ei in pairs(node.directions) do
    ket_rho[i] = norm(ei, 0)
    ket_e[i] = (ei .. ei) - 2
    ket_e2[i] = 1 / 2 * (15 * (ei .. ei) ^ 2 - 55 * (ei .. ei) + 32)
    ket_j_x[i] = ei[1]
    ket_q_x[i] = 1 / 2 * (5 * (ei .. ei) - 13) * ei[1]
    ket_j_y[i] = ei[2]
    ket_q_y[i] = 1 / 2 * (5 * (ei .. ei) - 13) * ei[2]
    ket_j_z[i] = ei[3]
    ket_q_z[i] = 1 / 2 * (5 * (ei .. ei) - 13) * ei[3]
    ket_p_x_x[i] = 3 * ei[1] ^ 2 - (ei .. ei)
    ket_p_w_w[i] = ei[2] ^ 2 - ei[3] ^ 2
    ket_p_x_y[i] = ei[1] * ei[2]
    ket_p_y_z[i] = ei[2] * ei[3]
    ket_p_z_x[i] = ei[3] * ei[1]
    ket_m_x_y_z[i] = ei[1] * ei[2] * ei[3]
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
    ket_j_z,
    ket_q_z,
    ket_p_x_x,
    ket_p_w_w,
    ket_p_x_y,
    ket_p_y_z,
    ket_p_z_x,
    ket_m_x_y_z
  }

  -- Print each moment (components)
  print("Transformation matrix:")
  print(self.M)
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
    error("Some of the moments are not orthogonal, check their components")
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
  print(self.invM)

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
          if denominator > 0 then
            self.invM[i][j] = "1./" .. denominator .. "."
          else
            self.invM[i][j] = "-1./" .. (-denominator) .. "."
          end
        end
      end
    end
  end
  --]]
  print("Inverted Matrix, rational expression")
  print(self.invM)
end

-- generate a few registers for the equilibrium moments
function MRTDHumieresD3Q15_2002:genLocalVarsDef()
  return self.precision .. " E_eq, E2_eq, Qx_eq, Qy_eq, Qz_eq, Pxx_eq, Pww_eq, Pxy_eq, Pyz_eq, Pxz_eq, Mxyz_eq"
end

-- Define the dynamics of the model
dynDHumieresD3Q15 = DynamicsGenerator("single_phase", {DensityMoment, VelocityMoment})

--generate the computation of the equilibrium moments
function dynDHumieresD3Q15:genEquilibriumMoments()
  local rho = model.density_name
  local rho0 = model.rho0
  local We = model.We
  local Wej = model.Wej
  local J = vector({"Jx", "Jy", "Jz"})
  return {
    "E_eq = " .. (-1 * rho + 1 / rho0 * (J .. J)),
    "E2_eq = " .. (We * rho + Wej / rho0 * (J .. J)),
    "Qx_eq = " .. (-7 / 3 * J[1]),
    "Qy_eq = " .. (-7 / 3 * J[2]),
    "Qz_eq = " .. (-7 / 3 * J[3]),
    "Pxx_eq = " .. (1 / (3 * rho0) * (2 * J[1] ^ 2 - (J[2] ^ 2 + J[3] ^ 2))),
    "Pww_eq = " .. (1 / rho0 * (J[2] ^ 2 - J[3] ^ 2)),
    "Pxy_eq = " .. (1 / rho0 * J[1] * J[2]),
    "Pyz_eq = " .. (1 / rho0 * J[2] * J[3]),
    "Pxz_eq = " .. (1 / rho0 * J[1] * J[3]),
    "Mxyz_eq = " .. (0)
  }
end

--helper function to relax moments
function dynDHumieresD3Q15:relax(name, s)
  return name .. " = " .. (name - s * (name - (name .. "_eq")))
end

-- generate the relaxation of the moments
function dynDHumieresD3Q15:genRelaxationMoments()
  return {
    self:relax("E", model.s1.name),
    self:relax("E2", model.s2.name),
    self:relax("Qx", model.s4.name),
    self:relax("Qy", model.s4.name),
    self:relax("Qz", model.s4.name),
    self:relax("Pxx", model.s9.name),
    self:relax("Pww", model.s9.name),
    self:relax("Pxy", model.s11.name),
    self:relax("Pyz", model.s11.name),
    self:relax("Pxz", model.s11.name),
    self:relax("Mxyz", model.s14.name)
  }
end

-- generate the distribution functions from some density, velocity, gradient of velocity
function dynDHumieresD3Q15:genInitMoments()
  local rho = model.density_name
  local V = model.velocity_name
  local rho0 = model.rho0
  local We = model.We
  local Wej = model.Wej
  local J = vector({"Jx", "Jy", "Jz"})
  return {
    "Jx = " .. (rho * V[1]),
    "Jy = " .. (rho * V[2]),
    "Jz = " .. (rho * V[3]),
    "E = " .. (-1 * rho + 1 / rho0 * (J .. J)),
    "E2 = " .. (We * rho + Wej / rho0 * (J .. J)),
    "Qx = " .. (-7 / 3 * J[1]),
    "Qy = " .. (-7 / 3 * J[2]),
    "Qz = " .. (-7 / 3 * J[3]),
    "Pxx = " .. (1 / (3 * rho0) * (2 * J[1] ^ 2 - (J[2] ^ 2 + J[3] ^ 2))),
    "Pww = " .. (1 / rho0 * (J[2] ^ 2 - J[3] ^ 2)),
    "Pxy = " .. (1 / rho0 * J[1] * J[2]),
    "Pyz = " .. (1 / rho0 * J[2] * J[3]),
    "Pxz = " .. (1 / rho0 * J[1] * J[3]),
    "Mxyz = " .. (0)
  }
end

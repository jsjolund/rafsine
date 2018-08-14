-- This file defines various methods of initialising distribution functions at the beginning of the program
-- TODO
-- replace double by model.precision


-- This method method will create a double shear layer, using the following formula
-- if y<=1/2 ux = tanh(width(y-1/4)) 
-- else      ux = tanh(width(3/4-y)) 
-- uy = amplitude*sin(2PI(x+1/4))
DoubleShearLayer = class()
function DoubleShearLayer:_init (parameters)
  -- width parameter of the layers (not actual width!)
  self.width = parameters["width"] or 80
  -- amplitudes of the oscillation in uy
  self.amplitude = parameters["amplitude"] or 0.05
end

-- Function to generate the code that set the macroscopic moments
function DoubleShearLayer:genMoments()
  local V = model.velocity_name
  local X = model.position_name
  local S = model.size_names
  local x = X[1].."/double("..(S[1]-1)..")"
  local y = X[2].."/double("..(S[2]-1)..")"
  --local Ma = 0.07
  if node.D == 2 then
    return {
      V[1].." = ("..y.."<=0.5) ? "..V[1].." = "..Ma.."*tanh("..self.width.."*("..y.."-0.25))"..": "..V[1].." = "..Ma.."*tanh("..self.width.."*(0.75-"..y.."))",
      V[2].." = "..self.amplitude*Ma.." * sin(2*M_PI*("..x.."+0.25))"
    }
  else --node.D = 3 then
    local z = X[3].."/double("..(S[3]-1)..")"
    --local coefZ = "exp(-"..(10*((z-0.5)^2))..")"
    local coefZ = 1
    return {
      V[1].." = ("..y.."<=0.5) ? "..V[1].." = "..Ma.."*"..coefZ.."*tanh("..self.width.."*("..y.."-0.25))"..": "..V[1].." = "..Ma.."*"..coefZ.."*tanh("..self.width.."*(0.75-"..y.."))",
      V[2].." = "..self.amplitude*Ma.."*"..coefZ.." * sin(2*M_PI*("..x.."+0.25))",
      V[3].." = 0"
    }
  end
end

-- Generate the code to compute the density, velocity and velocity gradient
function DoubleShearLayer:genRhoVdV()
  local V = model.velocity_name
  local X = model.position_name
  local S = model.size_names
  local x = X[1].."/double("..(S[1]-1)..")"
  local y = X[2].."/double("..(S[2]-1)..")"
  return {
    rho = 1,
    vx = ternary(y.."<=0.5", Ma.."*tanh("..self.width.."*("..y.."-0.25))", Ma.."*tanh("..self.width.."*(0.75-"..y.."))"),
    vy = self.amplitude*Ma.." * sin(2*M_PI*("..x.."+0.25))",
    --[[
    DxVx = 0,
    DxVy = (2*"M_PI"*self.amplitude*Ma*"cos(2*M_PI*("..x.."+0.25))"),
    DyVx = ternary(y.."<=0.5",
       self.width*Ma.."*(1-tanh("..self.width.."*("..y.."-0.25))*tanh("..self.width.."*("..y.."-0.25)))",
      -self.width*Ma.."*(1-tanh("..self.width.."*(0.75-"..y..")*tanh("..self.width.."*(0.75-"..y.."))))"),
    DyVy = 0,
    --]]
  }
end

-- Initialise the velocity field with a Taylor-Green Vortex
-- The velocity field is defined as
--    u = A cos(a*2PI*x) sin(b*2PI*y) sin(c*2PI*z)
--    v = B sin(a*2PI*x) cos(b*2PI*y) sin(c*2PI*z)
--    w = C sin(a*2PI*x) sin(b*2PI*y) cos(c*2PI*z)
--       where x,y,z in [0;1]
TaylorGreenVortex = class()
function TaylorGreenVortex:_init (parameters)
  -- amplitude along x
  self.A = parameters["A"] or 0.05
  -- number of repetitions along x
  self.a = parameters["a"] or 1
  -- amplitude along y
  self.B = parameters["B"] or 0.05
  -- number of repetitions along y
  self.b = parameters["b"] or 1
  -- amplitude along z
  self.C = parameters["C"] or 0.05
  -- number of repetitions along z
  self.c = parameters["c"] or 1
end

-- Function to generate the code that set the macroscopic moments
function TaylorGreenVortex:genMoments()
  local V = model.velocity_name
  local X = model.position_name
  local S = model.size_names
  local x = X[1].."/double("..(S[1]-1)..")"
  local y = X[2].."/double("..(S[2]-1)..")"
  local a2PIx = self.a.."*2*M_PI*"..x
  local b2PIy = self.b.."*2*M_PI*"..y
  --local Ma = 0.07
  local initV
  if node.D == 2 then
    initV = {
      V[1].." = "..self.A.."*cos("..a2PIx..")*sin("..b2PIy..")",
      V[2].." = "..self.B.."*sin("..a2PIx..")*cos("..b2PIy..")"
    }
  else --node.D == 3
    local z = X[3].."/double("..(S[3]-1)..")"
    local c2PIz = self.c.."*2*M_PI*"..z
    initV = {
      V[1].." = "..self.A.."*cos("..a2PIx..")*sin("..b2PIy..")*sin("..c2PIz..")",
      V[2].." = "..self.B.."*sin("..a2PIx..")*cos("..b2PIy..")*sin("..c2PIz..")",
      V[3].." = "..self.C.."*sin("..a2PIx..")*sin("..b2PIy..")*cos("..c2PIz..")",
    }
  end
  return initV
end

-- Initialise the velocity field with a Kida Vortex
-- The velocity field is defined as
--    u = U0 sin(x) * ( cos(3y)cos(z) - cos(y)cos(3z) )
--    v = U0 sin(y) * ( cos(3z)cos(x) - cos(z)cos(3x) )
--    w = U0 sin(z) * ( cos(3x)cos(y) - cos(x)cos(3y) )
--       where x,y,z in [0;2*PI]
KidaVortex = class()
function KidaVortex:_init (parameters)
  -- amplitude
  self.U0 = parameters["U0"] or 0.05
end

-- Function to generate the code that set the macroscopic moments
function KidaVortex:genMoments()
  if node.D == 2 then
     error("Kida Vortex only possible in 3D")
  end
  local sin = function(x)
    return "sin("..x..")"
  end
  local cos = function(x)
    return "cos("..x..")"
  end
  local PI = "M_PI"
  local V = model.velocity_name
  local X = model.position_name
  local S = model.size_names
  local x = 2*PI*X[1].."/double("..(S[1]-1)..")"
  local y = 2*PI*X[2].."/double("..(S[2]-1)..")"
  local z = 2*PI*X[3].."/double("..(S[3]-1)..")"
  return {
    V[1].." = "..( self.U0 * sin(x) * ( cos(3*y)*cos(z) - cos(y)*cos(3*z) ) ),
    V[2].." = "..( self.U0 * sin(y) * ( cos(3*z)*cos(x) - cos(z)*cos(3*x) ) ),
    V[3].." = "..( self.U0 * sin(z) * ( cos(3*x)*cos(y) - cos(x)*cos(3*y) ) ),
  }
end

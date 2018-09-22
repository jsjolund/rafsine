package.path = package.path .. ";lua/?.lua"
require "problems/data_center/settings"
require "VoxelGeometry"

print("Time-step : " .. uc:N_to_s(1) .. " s")
print("Creating geometry of size "..nx.."*"..ny.."*"..nz)
vox = VoxelGeometry(nx, ny, nz)

-- CRAC characteristics
CRAC = {
  -- position of the corner with minimum coordinates
  min = {1.4, 5.0, 0.0},
  -- position of the corner with maximum coordinates
  max = {3.4, 6.0, 1.8},
  -- volume flow rate through CRAC unit in m^3/s
  Q = 2
}
-- dimension of the CRAC outlet
CRAC.outletSize = (CRAC.max[1] - CRAC.min[1]) * (CRAC.max[2] - CRAC.min[2])
-- speed at the CRAC outlet in lattice units
CRAC.V = uc:Q_to_Ulu(CRAC.Q, CRAC.outletSize)
print("Speed of the CRAC in LU:", CRAC.V)

-- Floor vents characteristics
Vents = {
  -- corner position
  origin = {1.8, 0.6, 0.0},
  -- size along x
  x = 1.2,
  -- size along y
  y = 3.0
}
-- compute the vents inlet velocity to match the volume flow rate of the CRAC
Vents.inletSize = Vents.x * Vents.y
Vents.V = uc:Q_to_Ulu(CRAC.Q, Vents.inletSize)
print("Speed of the floor vents in LU:", Vents.V)

-- Set domain boundary conditions
vox:addWallXmin()
vox:addWallXmax()
vox:addWallYmin()
vox:addWallYmax()
vox:addWallZmin()
vox:addWallZmax()

-- Set an inlet on the floor
vox:addQuadBC(
{
  origin = Vents.origin,
  dir1 = {Vents.x, 0, 0},
  dir2 = {0, Vents.y, 0},
  typeBC = "inlet",
  normal = {0, 0, 1},
  velocity = {0, 0, Vents.V},
  temperature = {
    type_ = "constant",
    value = 16
  },
  mode = "overwrite",
  name = "CRAC floor vent"
})

-- Add walls for the CRAC unit
vox:addQuadBC(
{
  origin = CRAC.min,
  dir1 = {CRAC.max[1] - CRAC.min[1], 0, 0},
  dir2 = {0, 0, CRAC.max[3] - CRAC.min[3]},
  typeBC = "wall",
  normal = {0, -1, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = CRAC.min,
  dir1 = {0, CRAC.max[2] - CRAC.min[2], 0},
  dir2 = {0, 0, CRAC.max[3] - CRAC.min[3]},
  typeBC = "wall",
  normal = {-1, 0, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {CRAC.max[1], CRAC.min[2], CRAC.min[3]},
  dir1 = {0, CRAC.max[2] - CRAC.min[2], 0},
  dir2 = {0, 0, CRAC.max[3] - CRAC.min[3]},
  typeBC = "wall",
  normal = {1, 0, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {CRAC.min[1], CRAC.min[2], CRAC.max[3]},
  dir1 = {CRAC.max[1] - CRAC.min[1], 0, 0},
  dir2 = {0, CRAC.max[2] - CRAC.min[2], 0},
  typeBC = "inlet",
  normal = {0, 0, 1},
  velocity = {0, 0, -CRAC.V},
  temperature = {type_ = "zeroGradient"},
  mode = "fill",
  name = "CRAC outlet"
})

-- Empty the inside of the CRAC
vox:makeHollow(
{
  min = CRAC.min,
  max = CRAC.max,
  faces = {ymax = true, zmin = true} -- faces to remove
})

-- Create some servers
-- Description of the servers
servers = {
  --origins and normal correspond to the server front
  -- bottom left row
  BL = {
    powers = {2, 1, 2, 0, 1},
    origin = {1.8, 0.6, 0.0},
    normal = {1, 0, 0}
  },
  -- top left row
  TL = {
    powers = {0, 1, 2, 0, 1},
    origin = {1.8, 0.6, 1.0},
    normal = {1, 0, 0}
  },
  -- bottom right row
  BR = {
    powers = {0, 2, 1, 0, 0},
    origin = {3.0, 0.6, 0.0},
    normal = {-1, 0, 0}
  },
  -- top right row
  TR = {
    powers = {4, 0, 1, 1, 1},
    origin = {3.0, 0.6, 1.0},
    normal = {-1, 0, 0}
  }
}

-- power to volume flow rate correspondance (given)
pow_to_Q = {[0] = 0.0, [1] = 0.3, [2] = 0.4, [4] = 0.6}

-- compute the corresponding servers inlet/outlet velocities from the power
-- and difference of temperature accross the server
speeds = {}
temperatures = {}
for row, details in pairs(servers) do
  speeds[row] = {}
  temperatures[row] = {}
  for i, P in ipairs(details.powers) do
    local Q = pow_to_Q[P]
    speeds[row][i] = uc:Q_to_Ulu(Q, 0.6 * 1.0)
    if Q > 0 then
      --temperatures[row][i] = 0
      --temperatures[row][i] = 5*P
      local nu = 1.511e-5
      local k = 2.570e-5
      local Pr = 0.713
      temperatures[row][i] = P * nu / (Q * k * Pr)
    else
      temperatures[row][i] = 0
    end
    print("velocity", speeds[row][i], "temperature", temperatures[row][i])
  end
end

-- add servers wall
-- servers on the left
vox:addQuadBC(
{
  origin = {0.8, 0.6, 0.0},
  dir1 = {1.0, 0.0, 0.0},
  dir2 = {0.0, 0.0, 2.0},
  typeBC = "wall",
  normal = {0, -1, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {0.8, 3.6, 0.0},
  dir1 = {1.0, 0.0, 0.0},
  dir2 = {0.0, 0.0, 2.0},
  typeBC = "wall",
  normal = {0, 1, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {0.8, 0.6, 2.0},
  dir1 = {1.0, 0.0, 0.0},
  dir2 = {0.0, 3.0, 0.0},
  typeBC = "wall",
  normal = {0, 0, 1},
  mode = "intersect"
})
-- servers on the right
vox:addQuadBC(
{
  origin = {3.0, 0.6, 0.0},
  dir1 = {1.0, 0.0, 0.0},
  dir2 = {0.0, 0.0, 2.0},
  typeBC = "wall",
  normal = {0, -1, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {3.0, 3.6, 0.0},
  dir1 = {1.0, 0.0, 0.0},
  dir2 = {0.0, 0.0, 2.0},
  typeBC = "wall",
  normal = {0, 1, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {3.0, 0.6, 2.0},
  dir1 = {1.0, 0.0, 0.0},
  dir2 = {0.0, 3.0, 0.0},
  typeBC = "wall",
  normal = {0, 0, 1},
  mode = "intersect"
})

--add BC for the servers
for row, details in pairs(servers) do
  for i, V in ipairs(speeds[row]) do
    local n = vector(details.normal)
    -- if velocity is zero, its a wall, otherwise, its an inlet
    if V == 0 then
      typeBC = "wall"
    else
      typeBC = "inlet"
    end
    -- face facing the floor vents
    vox:addQuadBC(
    {
      origin = vector(details.origin) + vector({0.0, (i - 1) * 0.6, 0.0}),
      dir1 = {0.0, 0.6, 0.0},
      dir2 = {0.0, 0.0, 1.0},
      typeBC = typeBC,
      normal = n,
      velocity = {-n[1] * V, 0, 0},
      temperature = {type_ = "zeroGradient"},
      mode = "fill",
      name = "server "..row..i.." inlet, power " .. details.powers[i] .. " kW",
    })
    -- face facing the wall
    vox:addQuadBC(
    {
      origin = vector(details.origin) + vector({-1.0 * n[1], (i - 1) * 0.6, 0.0}),
      dir1 = {0.0, 0.6, 0.0},
      dir2 = {0.0, 0.0, 1.0},
      typeBC = typeBC,
      normal = -1 * n,
      velocity = {-n[1] * V, 0, 0},
      temperature = {
        type_ = "relative",
        value = temperatures[row][i],
        -- relative position (in m) of the reference BC
        rel_pos = 1.0
      },
      mode = "fill",
      name = "server "..row..i.." outlet, power " .. details.powers[i] .. " kW",
    })
  end
end

-- Empty the inside of the servers
-- Left server
vox:makeHollow(
{
  min = {0.8, 0.6, 0.0},
  max = {1.8, 3.6, 2.0},
  faces = {zmin = true} -- faces to remove
})
-- Right server
vox:makeHollow(
{
  min = {3.0, 0.6, 0.0},
  max = {4.0, 3.6, 2.0},
  faces = {zmin = true} -- faces to remove
})

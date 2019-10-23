package.path = package.path .. ";lua/?.lua"
require "problems/pod2/settings"
require "VoxelGeometry"
require "helpers"

print("Time-step : " .. uc:N_to_s(1) .. " s")
print("Creating geometry of size "..nx.."*"..ny.."*"..nz)
vox = VoxelGeometry(nx, ny, nz)

-- From SEE Cooler HDZ specs
cracX = 0.510
cracY = 1.225
cracZ = 2.55
cracOutletY = 1.00
cracOutletZoffset = 0.1
cracOutletZ = 1.875 - cracOutletZoffset

-- Server wall positions
lSrvWallX = 1.75
lSrvWallY = 0.0
lSrvWallZ = 0.0
rSrvWallX = 4.12
rSrvWallY = 0.0
rSrvWallZ = 0.0

-- Minkels server rack
rackX = 1.170
rackY = 0.800
rackZ = 2.220
rackInletWidth = 0.450

-- Rack unit in meters
U = 0.0445
num_srvs = 45
max_srvs = 45
srvZ = U * num_srvs
srvZmax = U * max_srvs
srvZoffset = 0.1

-- Length of a voxel in meters
C_L = uc:C_L()

-- CRAC characteristics. Positions are from POD 2 schematics
CRACs = {
  P02HDZ01 = {
    -- Position of the corner with minimum coordinates
    min = {mx - cracX, 0.6 + cracY + 0.45, 0},
    -- Position of the corner with maximum coordinates
    max = {mx - 0.0, 0.6 + cracY + 0.45 + cracY, cracZ},
    -- Volumetric flow rate through CRAC unit in m^3/s
    Q = 0.5,
    -- Temperature of output air
    T = 16.0,
    -- If 1, CRAC outlet along positive x axis, if -1 the negative
    facing = -1
  },
  P02HDZ02 = {
    min = {mx - cracX, 0.6, 0},
    max = {mx - 0.0, 0.6 + cracY, cracZ},
    Q = 0.5,
    T = 16.0,
    facing = -1
  },
  P02HDZ03 = {
    min = {0.0, 0.6, 0},
    max = {0 + cracX + 0.0, 0.6 + cracY, cracZ},
    Q = 0.5,
    T = 16.0,
    facing = 1
  },
  P02HDZ04 = {
    min = {0.0, 0.6 + cracY + 0.45, 0},
    max = {0 + cracX + 0.0, 0.6 + cracY + 0.45 + cracY, cracZ},
    Q = 0.5,
    T = 16.0,
    facing = 1
  }
}

-- Create servers
-- Description of the servers
servers = {
  -- Origins and normal correspond to the server front (intake)
  -- Right row
  P02R01 = {
    powers = {1},
    origin = {rSrvWallX, rSrvWallY + 4 * rackY, rSrvWallZ},
    normal = {1, 0, 0}
  },
  P02R02 = {
    powers = {4},
    origin = {rSrvWallX, rSrvWallY + 3 * rackY, rSrvWallZ},
    normal = {1, 0, 0}
  },
  P02R03 = {
    powers = {2},
    origin = {rSrvWallX, rSrvWallY + 2 * rackY, rSrvWallZ},
    normal = {1, 0, 0}
  },
  P02R04 = {
    powers = {1},
    origin = {rSrvWallX, rSrvWallY + 1 * rackY, rSrvWallZ},
    normal = {1, 0, 0}
  },
  P02R05 = {
    powers = {1},
    origin = {rSrvWallX, rSrvWallY + 0 * rackY, rSrvWallZ}, -- Closest to wall
    normal = {1, 0, 0}
  },
  -- Left row (closest to the door outside)
  P02R06 = {
    powers = {1},
    origin = {lSrvWallX, lSrvWallY + 0 * rackY, lSrvWallZ}, -- Closest to wall
    normal = {-1, 0, 0}
  },
  P02R07 = {
    powers = {2},
    origin = {lSrvWallX, lSrvWallY + 1 * rackY, lSrvWallZ},
    normal = {-1, 0, 0}
  },
  P02R08 = {
    powers = {1},
    origin = {lSrvWallX, lSrvWallY + 2 * rackY, lSrvWallZ},
    normal = {-1, 0, 0}
  },
  P02R09 = {
    powers = {4},
    origin = {lSrvWallX, lSrvWallY + 3 * rackY, lSrvWallZ},
    normal = {-1, 0, 0}
  },
  P02R10 = {
    powers = {1},
    origin = {lSrvWallX, lSrvWallY + 4 * rackY, lSrvWallZ},
    normal = {-1, 0, 0}
  }
}

-- Power to volume flow rate correspondance (given)
pow_to_Q = {[0] = 0.0, [1] = 0.083, [2] = 0.133, [4] = 0.221}

-- Set domain boundary conditions
vox:addWallXmin()
vox:addWallXmax()
vox:addWallYmin()
vox:addWallYmax()
vox:addWallZmin()
vox:addWallZmax()

-- Network rack
-- vox:addSolidBox({min = {mx - 0.6, my - 0.6 - 1.7, 0}, max = {mx, my - 1.7, 2.0}})

-- Generate CRAC geometry
for name, CRAC in pairs(CRACs) do
  -- Dimension of the CRAC outlet
  CRAC.outletSize = cracOutletY * cracOutletZ
  CRAC.inletSize = math.abs((CRAC.max[1] - CRAC.min[1] - C_L * 2) * (CRAC.max[2] - CRAC.min[2] - C_L * 2))
  CRAC.outletV = uc:Q_to_Ulu(CRAC.Q, CRAC.outletSize)
  CRAC.inletV = uc:Q_to_Ulu(CRAC.Q, CRAC.inletSize)
  print("Speed of " .. name .. " outlet in LU: " .. CRAC.outletV)
  print("Speed of " .. name .. " inlet in LU: " .. CRAC.inletV)

  vox:addQuadBC(
    {
      -- Front wall
      origin = {(CRAC.facing > 0) and CRAC.max[1] or CRAC.min[1], CRAC.min[2], CRAC.min[3]},
      dir1 = {0, CRAC.max[2] - CRAC.min[2], 0},
      dir2 = {0, 0, CRAC.max[3] - CRAC.min[3]},
      typeBC = "wall",
      normal = {CRAC.facing, 0, 0},
      mode = "intersect"
    })
  vox:addQuadBC(
    {
      -- Side wall
      origin = CRAC.min,
      dir1 = {CRAC.max[1] - CRAC.min[1], 0, 0},
      dir2 = {0, 0, CRAC.max[3] - CRAC.min[3]},
      typeBC = "wall",
      normal = {0, -1, 0},
      mode = "intersect"
    })
  vox:addQuadBC(
    {
      -- Other side wall
      origin = {CRAC.min[1], CRAC.max[2], CRAC.min[3]},
      dir1 = {CRAC.max[1] - CRAC.min[1], 0, 0},
      dir2 = {0, 0, CRAC.max[3] - CRAC.min[3]},
      typeBC = "wall",
      normal = {0, 1, 0},
      mode = "intersect"
    })
  vox:addQuadBC(
    {
      -- Top
      origin = {CRAC.min[1], CRAC.min[2], CRAC.max[3]},
      dir1 = {CRAC.max[1] - CRAC.min[1], 0, 0},
      dir2 = {0, CRAC.max[2] - CRAC.min[2], 0},
      typeBC = "wall",
      normal = {0, 1, 0},
      mode = "intersect"
    })

  -- Front outlet facing the room
  cracOutMin = {
    (CRAC.facing > 0) and CRAC.max[1] or CRAC.min[1],
    CRAC.min[2] + (cracY - cracOutletY) / 2,
    CRAC.min[3] + cracOutletZoffset
  }
  cracOutDir1 = {0, 0, cracOutletZ}
  cracOutDir2 = {0, cracOutletY, 0}
  cracOutMax = vector(cracOutMin) + vector(cracOutDir1) + vector(cracOutDir2)
  vox:addQuadBC({
    origin = cracOutMin,
    dir1 = cracOutDir1,
    dir2 = cracOutDir2,
    typeBC = "inlet",
    normal = {CRAC.facing, 0, 0},
    velocity = {CRAC.outletV * CRAC.facing, 0, 0},
    temperature = {
      type_ = "constant",
      value = CRAC.T
    },
    mode = "overwrite",
    name = name
  })
  vox:addSensor(
    {
      min = cracOutMin + vector({(CRAC.facing > 0) and C_L or -C_L, 0, 0}),
      max = cracOutMax + vector({(CRAC.facing > 0) and C_L or -C_L, 0, 0}),
      name = name .. "_out"
    })

  -- Inlet on top
  -- To make sure the correct extents are calculated we need to use 'overwrite'
  -- and offset the inlet by one voxel from the borders on top of the CRAC
  cracInMin = {CRAC.min[1] + C_L, CRAC.min[2] + C_L, CRAC.max[3]}
  cracInDir1 = {CRAC.max[1] - CRAC.min[1] - C_L * 2, 0, 0}
  cracInDir2 = {0, CRAC.max[2] - CRAC.min[2] - C_L * 2, 0}
  cracInMax = vector(cracInMin) + vector(cracInDir1) + vector(cracInDir2)
  vox:addQuadBC({
    origin = cracInMin,
    dir1 = cracInDir1,
    dir2 = cracInDir2,
    typeBC = "inlet",
    normal = {0, 0, 1},
    velocity = {0, 0, -CRAC.inletV},
    temperature = {type_ = "zeroGradient"},
    mode = "overwrite",
    name = name
  })
  vox:addSensor(
    {
      min = cracInMin + vector({0, 0, C_L}),
      max = cracInMax + vector({0, 0, C_L}),
      name = name .. "_in"
    })

  -- Empty the inside of the CRAC
  if (CRAC.facing > 0) then
    vox:makeHollow(
      {
        min = CRAC.min,
        max = CRAC.max,
        faces = {xmin = true, zmin = true} -- Faces to remove
      })
  else
    vox:makeHollow(
      {
        min = CRAC.min,
        max = CRAC.max,
        faces = {xmax = true, zmin = true} -- Faces to remove
      })
  end
end

-- Compute the corresponding servers inlet/outlet velocities from the power
-- and difference of temperature accross the server
speeds = {}
temperatures = {}
for name, rack in pairs(servers) do
  speeds[name] = {}
  temperatures[name] = {}
  for i, P in ipairs(rack.powers) do
    local Q = pow_to_Q[P]
    speeds[name][i] = uc:Q_to_Ulu(Q, rackX * rackY)
    if Q > 0 then
      temperatures[name][i] = P * nu / (Q * k * Pr)
    else
      temperatures[name][i] = 0
    end
    print("velocity", speeds[name][i], "temperature", temperatures[name][i])
  end
end

function addRackWall(params)
  vox:addQuadBC(
    {
      origin = {params.srvWallX, 0, 0},
      dir1 = {0, params.rackY*5, 0},
      dir2 = {0, 0, params.rackZ},
      typeBC = "wall",
      normal = {-1, 0, 0},
      mode = "intersect"
    })
  vox:addQuadBC(
    {
      origin = {params.srvWallX + params.rackX, 0, 0},
      dir1 = {0, params.rackY*5, 0},
      dir2 = {0, 0, params.rackZ},
      typeBC = "wall",
      normal = {1, 0, 0},
      mode = "intersect"
    })
  vox:addQuadBC(
    {
      origin = {params.srvWallX,
        params.srvWallY,
        params.srvWallZ + params.rackZ},
      dir1 = {params.rackX, 0, 0},
      dir2 = {0, params.rackY*5, 0},
      typeBC = "wall",
      normal = {0, 0, 1},
      mode = "intersect"
    })
  vox:addQuadBC(
    {
      origin = {params.srvWallX,
        params.srvWallY + params.rackY*5,
        params.srvWallZ },
      dir1 = {params.rackX, 0, 0},
      dir2 = {0, 0, params.rackZ},
      typeBC = "wall",
      normal = {0, 1, 0},
      mode = "intersect"
    })
end

addRackWall({
  srvWallX = lSrvWallX,
  srvWallY = lSrvWallY,
  srvWallZ = lSrvWallZ,
  rackX = rackX,
  rackY = rackY,
  rackZ = rackZ
})

addRackWall({
  srvWallX = rSrvWallX,
  srvWallY = rSrvWallY,
  srvWallZ = rSrvWallZ,
  rackX = rackX,
  rackY = rackY,
  rackZ = rackZ
})

-- Add BC for the servers
for name, rack in pairs(servers) do
  for i, V in ipairs(speeds[name]) do
    local n = vector(rack.normal)

    -- If velocity is zero, it's a wall, otherwise, it's an inlet
    local typeBC = "inlet"
    if V == 0 then
      typeBC = "wall"
    end

    -- Face facing the cold aisle
    vox:addQuadBC({
      origin = vector(rack.origin) +
      vector(
        {
          (n[1] < 0) and 0.0 or rackX,
          (rackY - rackInletWidth) / 2,
          (i - 1) * srvZ + srvZoffset
        }
      ),
      dir1 = {0.0, 0.0, srvZ},
      dir2 = {0.0, rackInletWidth, 0.0},
      typeBC = typeBC,
      normal = n,
      velocity = {-n[1] * V, 0, 0},
      temperature = {type_ = "zeroGradient"},
      mode = "overwrite",
      name = name
    })

    -- Face facing hot aisle
    vox:addQuadBC({
      origin = vector(rack.origin) +
      vector(
        {
          (n[1] > 0) and 0.0 or rackX,
          (rackY - rackInletWidth) / 2,
          (i - 1) * srvZ + srvZoffset
        }
      ),
      dir1 = {0.0, 0.0, srvZ},
      dir2 = {0.0, rackInletWidth, 0.0},
      typeBC = typeBC,
      normal = -n,
      velocity = {-n[1] * V, 0, 0},
      temperature = {
        type_ = "relative",
        value = temperatures[name][i],
        -- relative position (in m) of the reference BC
        rel_pos = rackX
      },
      mode = "overwrite",
      name = name
    })

    -- Add rack sensor strips
    local sensorInX = rack.origin[1] + ((n[1] < 0) and (-C_L) or (rackX + C_L))
    local sensorOutX = rack.origin[1] + ((n[1] > 0) and (-C_L) or (rackX + C_L))
    local sensorY = rack.origin[2] + (rackY / 2)
    local sensorsZ = {
      rack.origin[3] + srvZoffset + C_L,
      rack.origin[3] + srvZoffset + srvZ / 2,
      rack.origin[3] + srvZoffset + srvZ - C_L
    }
    local sensorsLoc = {"b", "m", "t"}
    for zi = 1, 3 do
      local sensorZ = sensorsZ[zi]
      local sensorLoc = sensorsLoc[zi]
      vox:addSensor(
      {
        min = {sensorInX, sensorY, sensorZ},
        max = {sensorInX, sensorY, sensorZ},
        name = name .. "_in_" .. sensorLoc
      })
      vox:addSensor(
      {
        min = {sensorOutX, sensorY, sensorZ},
        max = {sensorOutX, sensorY, sensorZ},
        name = name .. "_out_" .. sensorLoc
      })
    end

  end
end

-- Empty the inside of the servers
-- Left server rack
vox:makeHollow(
  {
    min = {lSrvWallX, lSrvWallY, lSrvWallZ},
    max = {lSrvWallX + rackX, lSrvWallY + 5 * rackY, lSrvWallZ + rackZ},
    faces = {ymin = true, zmin = true} -- faces to remove
  })

-- Right server rack
vox:makeHollow(
  {
    min = {rSrvWallX, rSrvWallY, rSrvWallZ},
    max = {rSrvWallX + rackX, rSrvWallY + 5 * rackY, rSrvWallZ + rackZ},
    faces = {ymin = true, zmin = true} -- faces to remove
  })

-- Door to servers
vox:addSolidBox(
  {
    min = {
      lSrvWallX + rackX,
      lSrvWallY + 5 * rackY - 0.05,
      0.0
    },
    max = {
      rSrvWallX,
      rSrvWallY + 5 * rackY,
      rackZ
    }
  })

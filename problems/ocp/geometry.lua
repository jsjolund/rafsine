package.path = package.path .. ";lua/?.lua"
require "problems/ocp/settings"
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
lSrvWallX = 1.79
lSrvWallY = 0.0
lSrvWallZ = 0.0
rSrvWallX = 4.17
rSrvWallY = 0.0
rSrvWallZ = 0.0

-- Rack unit in meters
U = 0.05

-- Minkels server rack
rackX = 1.08
rackY = 0.62
rackZ = 2.22
rackInletWidth = 0.45
srvY = rackInletWidth / 3
srvZ = 2*U

srvZoffset = 0.3 -- Height above floor

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

chassiZ = {
  0.3 + 0 * srvZ,
  0.3 + 1 * srvZ,
  0.3 + 2 * srvZ,
  0.7 + 0 * srvZ,
  0.7 + 1 * srvZ,
  1.17 + 0 * srvZ,
  1.17 + 1 * srvZ,
  1.17 + 2 * srvZ,
  1.57 + 0 * srvZ,
  1.57 + 1 * srvZ,
}
servers = {}
for rack=1,6 do
  for chassi=1,10 do
    name = "P02R"..string.format("%02d",7-rack).."C"..string.format("%02d",chassi)
    servers[name] =
    {
      powers = {1},
      origin = {rSrvWallX, 
                rSrvWallY + (rack - 1) * rackY, 
                rSrvWallZ + chassiZ[chassi]},
      normal = {1, 0, 0}
    }
  end
end
for rack=1,6 do
  for chassi=1,10 do
    name = "P02R"..string.format("%02d",6+rack).."C"..string.format("%02d",chassi)
    servers[name] =
    {
      powers = {1},
      origin = {lSrvWallX, 
                lSrvWallY + (rack - 1) * rackY, 
                lSrvWallZ + chassiZ[chassi]},
      normal = {-1, 0, 0}
    }
  end
end

-- Power to volume flow rate correspondance (given)
pow_to_Q = {[0] = 0.0, [1] = 0.083, [2] = 0.133, [4] = 0.221}

-- Set domain boundary conditions
vox:addWallXmin()
vox:addWallXmax()
vox:addWallYmin()
vox:addWallYmax()
vox:addWallZmin()
vox:addWallZmax()

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
      dir1 = {0, params.rackY*6, 0},
      dir2 = {0, 0, params.rackZ},
      typeBC = "wall",
      normal = {-1, 0, 0},
      mode = "intersect"
    })
  vox:addQuadBC(
    {
      origin = {params.srvWallX + params.rackX, 0, 0},
      dir1 = {0, params.rackY*6, 0},
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
      dir2 = {0, params.rackY*6, 0},
      typeBC = "wall",
      normal = {0, 0, 1},
      mode = "intersect"
    })
  -- vox:addQuadBC(
  --   {
  --     origin = {params.srvWallX,
  --       params.srvWallY + params.rackY*6,
  --       params.srvWallZ },
  --     dir1 = {params.rackX, 0, 0},
  --     dir2 = {0, 0, params.rackZ},
  --     typeBC = "wall",
  --     normal = {0, 1, 0},
  --     mode = "intersect"
  --   })
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
for name, chassi in pairs(servers) do
  for i, V in ipairs(speeds[name]) do
    local n = vector(chassi.normal)

    -- If velocity is zero, it's a wall, otherwise, it's an inlet
    local typeBC = "inlet"
    if V == 0 then
      typeBC = "wall"
    end

    for srv=1,3 do
      srvName = (n[1] > 0) and name.."SRV0"..srv or name.."SRV0"..(4-srv)
      -- Face facing the cold aisle
      vox:addQuadBC({
        origin = vector(chassi.origin) +
        vector(
          {
            (n[1] < 0) and 0.0 or rackX,
            (rackY - rackInletWidth) / 2 + srvY*(srv - 1),
            0
          }
        ),
        dir1 = {0.0, 0.0, srvZ},
        dir2 = {0.0, srvY, 0.0},
        typeBC = typeBC,
        normal = n,
        velocity = {-n[1] * V, 0, 0},
        temperature = {type_ = "zeroGradient"},
        mode = "overwrite",
        name = srvName
      })

      -- Face facing hot aisle
      vox:addQuadBC({
        origin = vector(chassi.origin) +
        vector(
          {
            (n[1] > 0) and 0.0 or rackX,
            (rackY - rackInletWidth) / 2 + srvY*(srv - 1),
            0
          }
        ),
        dir1 = {0.0, 0.0, srvZ},
        dir2 = {0.0, srvY, 0.0},
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
        name = srvName
      })
    end

    -- -- Add rack sensor strips
    -- local sensorInX = rack.origin[1] + ((n[1] < 0) and (-C_L) or (rackX + C_L))
    -- local sensorOutX = rack.origin[1] + ((n[1] > 0) and (-C_L) or (rackX + C_L))
    -- local sensorY = rack.origin[2] + (rackY / 2)
    -- local sensorsZ = {
    --   rack.origin[3] + srvZoffset + C_L,
    --   rack.origin[3] + srvZoffset + srvZ / 2,
    --   rack.origin[3] + srvZoffset + srvZ - C_L
    -- }
    -- local sensorsLoc = {"b", "m", "t"}
    -- for zi = 1, 3 do
    --   local sensorZ = sensorsZ[zi]
    --   local sensorLoc = sensorsLoc[zi]
    --   vox:addSensor(
    --   {
    --     min = {sensorInX, sensorY, sensorZ},
    --     max = {sensorInX, sensorY, sensorZ},
    --     name = name .. "_in_" .. sensorLoc
    --   })
    --   vox:addSensor(
    --   {
    --     min = {sensorOutX, sensorY, sensorZ},
    --     max = {sensorOutX, sensorY, sensorZ},
    --     name = name .. "_out_" .. sensorLoc
    --   })
    -- end

  end
end

vox:addSolidBox(
  {
    min = {
      0,
      lSrvWallY + 6 * rackY,
      0.0
    },
    max = {
      mx,
      my,
      2.25
    }
  })

-- Empty the inside of the servers
-- Left server rack
vox:makeHollow(
  {
    min = {lSrvWallX, lSrvWallY, lSrvWallZ},
    max = {lSrvWallX + rackX, lSrvWallY + 6 * rackY, lSrvWallZ + rackZ},
    faces = {ymin = true, zmin = true, ymax = true} -- faces to remove
  })

-- Right server rack
vox:makeHollow(
  {
    min = {rSrvWallX, rSrvWallY, rSrvWallZ},
    max = {rSrvWallX + rackX, rSrvWallY + 6 * rackY, rSrvWallZ + rackZ},
    faces = {ymin = true, zmin = true, ymax = true} -- faces to remove
  })

vox:makeHollow(
  {
    min = {
      0,
      lSrvWallY + 6 * rackY,
      0.0
    },
    max = {
      mx,
      my,
      2.25
    },
    faces = {xmin = true, xmax = true, ymax = true, zmin = true} -- faces to remove
  })

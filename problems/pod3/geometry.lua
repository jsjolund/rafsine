package.path = package.path .. ";lua/?.lua"
require "problems/pod3/settings"
require "VoxelGeometry"

print("Time-step : "..uc:N_to_s(1).." s")
print("Voxel size : "..uc:C_L().." m")
print("Creating geometry of size "..nx.."*"..ny.."*"..nz)
vox = VoxelGeometry(nx, ny, nz)

-- Length of a voxel in meters
C_L = uc:C_L()

-- Set domain boundary conditions
vox:addWallXmin()
vox:addWallXmax()
vox:addWallYmin()
vox:addWallYmax()
vox:addWallZmin()
vox:addWallZmax()

-- Power to volume flow rate correspondance (given)
pow_to_Q = {[0] = 0, [50] = 0.010, [60] = 0.015, [70] = 0.020}
-- pow_to_Q = {[0] = 0, [50] = 0, [60] = 0, [70] = 0}

-- Rack unit in meters
U = 0.05
rackInletWidth = 0.45
srvY = rackInletWidth / 3
srvZ = 2*U

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

doorX = 0.87

rackY = 0.6
rackX = 1.0
rackZ = 2.2

lSrvRowX = 1.0
lSrvRowY = 0.9
lSrvRowZ = 0

rSrvRowX = lSrvRowX + rackX + doorX
rSrvRowY = lSrvRowY
rSrvRowZ = lSrvRowZ

servers = {}
for rack=1,6 do
  for chassi=1,10 do
    name = "P02R"..string.format("%02d",7-rack).."C"..string.format("%02d",chassi)
    servers[name] =
    {
      powers = {70},
      origin = {rSrvRowX, 
                rSrvRowY + (rack - 1) * rackY, 
                rSrvRowZ + chassiZ[chassi]},
      normal = {1, 0, 0}
    }
  end
end
for rack=1,6 do
  for chassi=1,10 do
    name = "P02R"..string.format("%02d",6+rack).."C"..string.format("%02d",chassi)
    servers[name] =
    {
      powers = {70},
      origin = {lSrvRowX, 
                lSrvRowY + (rack - 1) * rackY, 
                lSrvRowZ + chassiZ[chassi]},
      normal = {-1, 0, 0}
    }
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
    speeds[name][i] = uc:Q_to_Ulu(Q, srvY * srvZ)
    -- Thermal conductivity
    k = 2.624e-5
    -- Prandtl number of air
    Pr = 0.707
    if Q > 0 then
      temperatures[name][i] = P * nu / (1000 * Q * k * Pr)
    else
      temperatures[name][i] = 0
    end
    print("velocity", speeds[name][i], "temperature", temperatures[name][i])
  end
end

function addRackWall(params)
  vox:addQuadBC(
    {
      origin = {params.srvRowX, 
                params.srvRowY, 
                params.srvRowZ},
      dir1 = {0, params.rackY*6, 0},
      dir2 = {0, 0, params.rackZ},
      typeBC = "wall",
      normal = {-1, 0, 0},
      mode = "intersect"
    })
  vox:addQuadBC(
    {
      origin = {params.srvRowX + params.rackX, 
                params.srvRowY, 
                params.srvRowZ},
      dir1 = {0, params.rackY*6, 0},
      dir2 = {0, 0, params.rackZ},
      typeBC = "wall",
      normal = {1, 0, 0},
      mode = "intersect"
    })
  vox:addQuadBC(
    {
      origin = {params.srvRowX,
                params.srvRowY,
                params.srvRowZ + params.rackZ},
      dir1 = {params.rackX, 0, 0},
      dir2 = {0, params.rackY*6, 0},
      typeBC = "wall",
      normal = {0, 0, 1},
      mode = "intersect"
    })
  vox:addQuadBC(
    {
      origin = {params.srvRowX,
                params.srvRowY + params.rackY*6,
                params.srvRowZ },
      dir1 = {params.rackX, 0, 0},
      dir2 = {0, 0, params.rackZ},
      typeBC = "wall",
      normal = {0, 1, 0},
      mode = "intersect"
    })
  vox:addQuadBC(
    {
      origin = {params.srvRowX,
                params.srvRowY,
                params.srvRowZ },
      dir1 = {params.rackX, 0, 0},
      dir2 = {0, 0, params.rackZ},
      typeBC = "wall",
      normal = {0, -1, 0},
      mode = "intersect"
    })
  vox:makeHollow(
    {
      min = {params.srvRowX, params.srvRowY, params.srvRowZ},
      max = {params.srvRowX + params.rackX,
            params.srvRowY + params.rackY*6,
            params.srvRowZ + params.rackZ},
      faces = {zmin = true}
    })
end

addRackWall({
  srvRowX = lSrvRowX,
  srvRowY = lSrvRowY,
  srvRowZ = lSrvRowZ,
  rackX = rackX,
  rackY = rackY,
  rackZ = rackZ
})

addRackWall({
  srvRowX = rSrvRowX,
  srvRowY = rSrvRowY,
  srvRowZ = rSrvRowZ,
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
            (n[1] < 0) and 0 or rackX,
            (rackY - rackInletWidth) / 2 + srvY*(srv - 1),
            0
          }
        ),
        dir1 = {0, 0, srvZ},
        dir2 = {0, srvY, 0},
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
            (n[1] > 0) and 0 or rackX,
            (rackY - rackInletWidth) / 2 + srvY*(srv - 1),
            0
          }
        ),
        dir1 = {0, 0, srvZ},
        dir2 = {0, srvY, 0},
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
  end
end

-- Add CRACs
cracX = 1.37
cracY = 0.92
cracZ = 2.40
cracUpperXY = 0.7

cracOutO = 0.07
cracOutX = 1.21
cracOutZ = 0.44
cracInXY = 0.70

cracQ = 1.0
cracT = 10.0

cracOutletSize = cracOutX * cracOutZ
cracInletSize = cracInXY * cracInXY
cracOutletV = uc:Q_to_Ulu(cracQ, cracOutletSize)
cracInletV = uc:Q_to_Ulu(cracQ, cracInletSize)

-- Left CRAC
vox:addSolidBox(
{
  min = {0, my - cracY, 0},
  max = {cracX, my, cracZ}
})
vox:addSolidBox(
{
  min = {cracX - cracUpperXY - (cracX - cracUpperXY)/2, my - cracUpperXY, cracZ},
  max = {cracX - (cracX - cracUpperXY)/2, my, mz}
})
vox:makeHollow(
{
  min = {cracX - cracUpperXY - (cracX - cracUpperXY)/2, my - cracUpperXY, cracZ},
  max = {cracX - (cracX - cracUpperXY)/2, my, mz},
  faces = {ymax = true, zmax = true, zmin = true}
})
-- Left CRAC air out
vox:addQuadBC({
  origin = {cracOutO, my - cracY, cracOutO},
  dir1 = {cracOutX, 0, 0},
  dir2 = {0, 0, cracOutZ},
  typeBC = "inlet",
  normal = {0, -1, 0},
  velocity = {0, -cracOutletV, 0},
  temperature = {
    type_ = "constant",
    value = cracT
  },
  mode = "overwrite",
  name = "CRAC_left"
})
-- Left CRAC air in
vox:addQuadBC({
  origin = {lSrvRowX + rackX, lSrvRowY, mz},
  dir1 = {cracInXY, 0, 0},
  dir2 = {0, cracInXY, 0},
  typeBC = "inlet",
  normal = {0, 0, -1},
  velocity = {0, 0, cracInletV},
  temperature = {
    type_ = "zeroGradient"
  },
  mode = "overwrite",
  name = "CRAC_left"
})

-- Right CRAC
vox:addSolidBox(
{
  min = {mx - cracX, my - cracY, 0},
  max = {mx, my, cracZ}
})
vox:addSolidBox(
{
  min = {mx - cracX + (cracX - cracUpperXY)/2, my - cracUpperXY, cracZ},
  max = {mx - (cracX - cracUpperXY)/2, my, mz}
})
vox:makeHollow(
{
  min = {mx - cracX + (cracX - cracUpperXY)/2, my - cracUpperXY, cracZ},
  max = {mx - (cracX - cracUpperXY)/2, my, mz},
  faces = {ymax = true, zmax = true, zmin = true}
})
-- Right CRAC air out
vox:addQuadBC({
  origin = {mx - cracX + cracOutO, my - cracY, cracOutO},
  dir1 = {cracOutX, 0, 0},
  dir2 = {0, 0, cracOutZ},
  typeBC = "inlet",
  normal = {0, -1, 0},
  velocity = {0, -cracOutletV, 0},
  temperature = {
    type_ = "constant",
    value = cracT
  },
  mode = "overwrite",
  name = "CRAC_right"
})
-- Right CRAC air in
vox:addQuadBC({
  origin = {lSrvRowX + rackX, lSrvRowY + cracInXY + 0.4, mz},
  dir1 = {cracInXY, 0, 0},
  dir2 = {0, cracInXY, 0},
  typeBC = "inlet",
  normal = {0, 0, -1},
  velocity = {0, 0, cracInletV},
  temperature = {
    type_ = "zeroGradient"
  },
  mode = "overwrite",
  name = "CRAC_right"
})

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
pow_to_Q = {[50] = 0.008, [60] = 0.010, [70] = 0.012}
pow_to_Q_keys = {50, 60, 70}

-- Rack and server measurements
rackInletWidth = 0.45
srvY = rackInletWidth / 3
-- Rack unit in meters
U = 0.05
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

listOffset = 0.1

-- CRAC measurements
cracX = 1.37
cracY = 0.92
cracZ = 2.40
cracUpperXY = 0.7

cracOutO = 0.07
cracOutX = 1.21
cracOutZ = 0.44
cracInXY = 0.70

cracQ = 2.0
cracT = 23.0

cracOutletSize = cracOutX * cracOutZ
cracInletSize = cracInXY * cracInXY
cracOutletV = uc:Q_to_Ulu(cracQ, cracOutletSize)
cracInletV = uc:Q_to_Ulu(cracQ, cracInletSize)

-- Temperature strips
sensorStripZ = {}
sensorStripZ["b"] = {origin = {0.4}}
sensorStripZ["m"] = {origin = {1.2}}
sensorStripZ["t"] = {origin = {2.0}}

sensorStripXY = {}
sensorStripXY["sensors_racks_01_to_03_in_"] = {
  origin = {rSrvRowX + rackX + C_L,
            rSrvRowY + 4*rackY}
}
sensorStripXY["sensors_racks_01_to_03_out_"] = {
  origin = {rSrvRowX - listOffset,
            rSrvRowY + 4*rackY}
}
sensorStripXY["sensors_racks_04_to_06_in_"] = {
  origin = {rSrvRowX + rackX + C_L,
            rSrvRowY + 1*rackY}
}
sensorStripXY["sensors_racks_04_to_06_out_"] = {
  origin = {rSrvRowX - listOffset,
            rSrvRowY + 1*rackY}
}
sensorStripXY["sensors_racks_07_to_09_in_"] = {
  origin = {lSrvRowX - C_L,
            lSrvRowY + 1*rackY}
}
sensorStripXY["sensors_racks_07_to_09_out_"] = {
  origin = {lSrvRowX + rackX + listOffset,
            lSrvRowY + 1*rackY}
}
sensorStripXY["sensors_racks_10_to_12_in_"] = {
  origin = {lSrvRowX - C_L,
            lSrvRowY + 4*rackY}
}
sensorStripXY["sensors_racks_10_to_12_out_"] = {
  origin = {lSrvRowX + rackX + listOffset,
            lSrvRowY + 4*rackY}
}
for namePrefix, posXY in pairs(sensorStripXY) do
  for posName, posZ in pairs(sensorStripZ) do
    name = namePrefix..posName
    x = posXY.origin[1]
    y = posXY.origin[2]
    z = posZ.origin[1]
    print("Adding sensor "..name.." at x="..x..", y="..y..", z="..z)
    vox:addSensor(
    {
      min = {x, y, z},
      max = {x, y, z},
      name = name
    })
  end
end

-- Create map of server boundary conditions
servers = {}
for rack=1,6 do
  for chassi=1,10 do
    name = "P02R"..string.format("%02d",7-rack).."C"..string.format("%02d",chassi)
    servers[name] =
    {
      powers = {pow_to_Q_keys[math.random(1,3)]},
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
      powers = {pow_to_Q_keys[math.random(1,3)]},
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

-- Left server racks
-- Back exhaust
vox:addQuadBC(
{
  origin = {lSrvRowX + rackX,
            lSrvRowY + C_L,
            lSrvRowZ},
  dir1 = {0, rackY*6 - 2*C_L, 0},
  dir2 = {0, 0, rackZ},
  typeBC = "wall",
  normal = {1, 0, 0},
  mode = "intersect"
})
-- Top
vox:addQuadBC(
{
  origin = {lSrvRowX + C_L,
            lSrvRowY + C_L,
            lSrvRowZ + rackZ},
  dir1 = {rackX - C_L, 0, 0},
  dir2 = {0, rackY*6 - 2*C_L, 0},
  typeBC = "wall",
  normal = {0, 0, 1},
  mode = "intersect"
})

-- Right server racks
-- Back exhaust
vox:addQuadBC(
{
  origin = {rSrvRowX,
            rSrvRowY + C_L,
            rSrvRowZ},
  dir1 = {0, rackY*6 - 2*C_L, 0},
  dir2 = {0, 0, rackZ},
  typeBC = "wall",
  normal = {-1, 0, 0},
  mode = "intersect"
})
-- Top
vox:addQuadBC(
{
  origin = {rSrvRowX,
            rSrvRowY + C_L,
            rSrvRowZ + rackZ},
  dir1 = {rackX - C_L, 0, 0},
  dir2 = {0, rackY*6 - 2*C_L, 0},
  typeBC = "wall",
  normal = {0, 0, 1},
  mode = "intersect"
})

-- Add rack containment
vox:addQuadBC(
{
  origin = {lSrvRowX,
            lSrvRowY,
            lSrvRowZ},
  dir1 = {0, rackY*6, 0},
  dir2 = {0, 0, mz - lSrvRowZ},
  typeBC = "wall",
  normal = {-1, 0, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {lSrvRowX + C_L,
            lSrvRowY + C_L,
            lSrvRowZ},
  dir1 = {0, rackY*6 - C_L*2, 0},
  dir2 = {0, 0, mz - lSrvRowZ},
  typeBC = "wall",
  normal = {1, 0, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {rSrvRowX + rackX,
            rSrvRowY,
            rSrvRowZ},
  dir1 = {0, rackY*6, 0},
  dir2 = {0, 0, mz - rSrvRowZ},
  typeBC = "wall",
  normal = {1, 0, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {rSrvRowX + rackX - C_L,
            rSrvRowY + C_L,
            rSrvRowZ},
  dir1 = {0, rackY*6 - C_L*2, 0},
  dir2 = {0, 0, mz - lSrvRowZ},
  typeBC = "wall",
  normal = {-1, 0, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {lSrvRowX,
            lSrvRowY,
            lSrvRowZ},
  dir1 = {rackX*2 + doorX, 0, 0},
  dir2 = {0, 0, mz - lSrvRowZ},
  typeBC = "wall",
  normal = {0, -1, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {lSrvRowX + C_L,
            lSrvRowY + C_L,
            lSrvRowZ},
  dir1 = {rackX*2 + doorX - C_L*2, 0, 0},
  dir2 = {0, 0, mz - lSrvRowZ},
  typeBC = "wall",
  normal = {0, 1, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {lSrvRowX,
            lSrvRowY + rackY*6,
            lSrvRowZ},
  dir1 = {rackX*2 + doorX, 0, 0},
  dir2 = {0, 0, mz - lSrvRowZ},
  typeBC = "wall",
  normal = {0, 1, 0},
  mode = "intersect"
})
vox:addQuadBC(
{
  origin = {lSrvRowX + C_L,
            lSrvRowY + rackY*6 - C_L,
            lSrvRowZ},
  dir1 = {rackX*2 + doorX - C_L*2, 0, 0},
  dir2 = {0, 0, mz - lSrvRowZ},
  typeBC = "wall",
  normal = {0, -1, 0},
  mode = "intersect"
})

-- Hollow inside racks
vox:makeHollow(
{
  min = {lSrvRowX, lSrvRowY, lSrvRowZ},
  max = {lSrvRowX + rackX,
        lSrvRowY + rackY*6,
        lSrvRowZ + rackZ},
  faces = {zmin = true}
})
vox:makeHollow(
{
  min = {rSrvRowX, rSrvRowY, rSrvRowZ},
  max = {rSrvRowX + rackX,
        rSrvRowY + rackY*6,
        rSrvRowZ + rackZ},
  faces = {zmin = true}
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
        vector({
          (n[1] < 0) and 0 or rackX,
          (rackY - rackInletWidth) / 2 + srvY*(srv - 1),
          0
        }),
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
        vector({
          (n[1] > 0) and 0 or rackX,
          (rackY - rackInletWidth) / 2 + srvY*(srv - 1),
          0
        }),
        dir1 = {0, 0, srvZ},
        dir2 = {0, srvY, 0},
        typeBC = typeBC,
        normal = -n,
        velocity = {-n[1] * V, 0, 0},
        temperature = {
          type_ = "relative",
          value = temperatures[name][i],
          rel_pos = rackX
        },
        mode = "overwrite",
        name = srvName
      })
    end
  end
end

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
lCracOutMin = {cracOutO, my - cracY, cracOutO}
lCracOutMax = lCracOutMin + vector({cracOutX, 0, cracOutZ})
vox:addQuadBC({
  origin = lCracOutMin,
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
vox:addSensor(
{
  min = lCracOutMin + vector({0, -C_L, 0}),
  max = lCracOutMax + vector({0, -C_L, 0}),
  name = "CRAC_left_out"
})
-- Left CRAC air in
lCracInMin = {lSrvRowX + rackX, lSrvRowY + C_L*2, mz}
lCracInMax = lCracInMin + vector({cracInXY, cracInXY, 0})
vox:addQuadBC({
  origin = lCracInMin,
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
vox:addSensor(
{
  min = lCracInMin + vector({0, 0, -C_L}),
  max = lCracInMax + vector({0, 0, -C_L}),
  name = "CRAC_left_in"
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
rCracOutMin = {mx - cracX + cracOutO, my - cracY, cracOutO}
rCracOutMax = rCracOutMin + vector({cracOutX, 0, cracOutZ})
vox:addQuadBC({
  origin = rCracOutMin,
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
vox:addSensor(
{
  min = rCracOutMin + vector({0, -C_L, 0}),
  max = rCracOutMax + vector({0, -C_L, 0}),
  name = "CRAC_right_out"
})
-- Right CRAC air in
rCracInMin = {lSrvRowX + rackX, lSrvRowY + C_L*2 + cracInXY + 0.4, mz}
rCracInMax = rCracInMin + vector({cracInXY, cracInXY, 0})
vox:addQuadBC({
  origin = rCracInMin,
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
vox:addSensor(
{
  min = rCracInMin + vector({0, 0, -C_L}),
  max = rCracInMax + vector({0, 0, -C_L}),
  name = "CRAC_right_in"
})

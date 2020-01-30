package.path = package.path .. "./?.lua;lua/?.lua"
require "problems/fan_chamber/settings"
require "VoxelGeometry"

print("Time-step : " .. uc:N_to_s(1) .. " s")
print("Creating geometry of size ", nx, ny, nz)
vox = VoxelGeometry(nx, ny, nz)

-- Length of a voxel in meters
C_L = uc:C_L()
print("C_L = "..C_L.." m")

-- Set domain boundary conditions
vox:addWallXmin()
vox:addWallXmax()
vox:addWallYmin()
vox:addWallYmax()
vox:addWallZmin()
vox:addWallZmax()

ventFlow = 5
fanFlow = 5
fanTemp = 20

boxSize = 2.5

fanSpeed = uc:Q_to_Ulu(fanFlow, (boxSize-C_L*2)*(boxSize-C_L*2))
ventSpeed = uc:Q_to_Ulu(ventFlow, (mx-C_L*2)*(mz-C_L*2))

vox:addQuadBC(
  {
    origin = {C_L, 0, C_L},
    dir1 = {mx-C_L*2, 0, 0},
    dir2 = {0, 0, mz-C_L*2},
    typeBC = "inlet",
    normal = {0, 1, 0},
    velocity = {0, ventSpeed, 0},
    temperature = {
      type_ = "constant", 
      value = 20
    },
    mode = "overwrite",
    name = "vent_input",
  })

vox:addQuadBC(
  {
    origin = {C_L, my, C_L},
    dir1 = {mx-C_L*2, 0, 0},
    dir2 = {0, 0, mz-C_L*2},
    typeBC = "inlet",
    normal = {0, -1, 0},
    velocity = {0, ventSpeed, 0},
    temperature = {type_ = "zeroGradient"},
    mode = "overwrite",
    name = "vent_output",
  })

vox:addSensor(
  {
    min = {C_L, C_L, C_L},
    max = {mx-C_L, C_L, mz-C_L},
    name = "vent_input_sensor"
  })

vox:addSensor(
  {
    min = {C_L, my-C_L, C_L},
    max = {mx-C_L, my-C_L, mz-C_L},
    name = "vent_output_sensor"
  })

fanMin = {
  mx/2 - boxSize/2,
  my/2 - boxSize/2,
  mz/2 - boxSize/2,
}
fanMax = {
  mx/2 + boxSize/2,
  my/2 + boxSize/2,
  mz/2 + boxSize/2,
}
vox:addSolidBox(
  {
    name = "fan",
    min = fanMin,
    max = fanMax,
  })

vox:addQuadBC(
  {
    origin = vector(fanMin) + vector({C_L, 0.0, C_L}),
    dir1 = {boxSize-2*C_L, 0.0, 0.0},
    dir2 = {0.0, 0.0, boxSize-2*C_L},
    typeBC = "inlet",
    normal = {0, -1, 0},
    velocity = {0, fanSpeed, 0},
    temperature = {type_ = "zeroGradient"},
    mode = "overwrite",
    name = "fan_input",
  })

vox:addQuadBC(
  {
    origin = vector({fanMin[1], fanMax[2], fanMin[3]}) + vector({C_L, 0.0, C_L}),
    dir1 = {boxSize-2*C_L, 0.0, 0.0},
    dir2 = {0.0, 0.0, boxSize-2*C_L},
    typeBC = "inlet",
    normal = {0, 1, 0},
    velocity = {0, fanSpeed, 0},
    temperature = {
      type_ = "relative",
      value = fanTemp,
      rel_pos = boxSize,
    },
    mode = "overwrite",
    name = "fan_output",
  })

vox:addSensor(
  {
    min = vector(fanMin) + vector({C_L, -C_L, C_L}),
    max = vector(fanMin) + vector({boxSize-C_L, -C_L, boxSize-C_L}),
    name = "fan_input_sensor"
  })

vox:addSensor(
  {
    min = vector({fanMin[1], fanMax[2], fanMin[3]}) + vector({C_L, C_L, C_L}),
    max = vector({fanMin[1], fanMax[2], fanMin[3]}) + vector({boxSize-C_L, C_L, boxSize-C_L}),
    name = "fan_output_sensor"
  })

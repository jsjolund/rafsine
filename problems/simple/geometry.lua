package.path = package.path .. "./?.lua;lua/?.lua"
require "problems/simple/settings"
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

ventSpeed = 0.25
ventSpeedLU = uc:ms_to_lu(ventSpeed)

expected_flow = ventSpeed * (mx - 2*C_L) * (my - 2*C_L)
print("Expected flow = "..expected_flow.." m3/s")

-- Set an inlet on one wall
vox:addQuadBC(
  {
    origin = {C_L, C_L, 0},
    dir1 = {mx - 2*C_L, 0, 0},
    dir2 = {0, my - 2*C_L, 0},
    typeBC = "inlet",
    normal = {0, 0, 1},
    velocity = {0, 0, ventSpeedLU},
    temperature = {
      type_ = "constant",
      value = 10
    },
    mode = "overwrite",
    name = "vent_bottom",
  })

vox:addSensor(
  {
    min = {C_L, C_L, C_L},
    max = {mx, my, 2*C_L},
    name = "vent_bottom_sensor"
  })

--Set an outlet on another wall
vox:addQuadBC(
  {
    origin = {C_L, C_L, mz},
    dir1 = {mx - 2*C_L, 0, 0},
    dir2 = {0, my - 2*C_L, 0},
    typeBC = "inlet",
    normal = {0, 0, -1},
    velocity = {0, 0, ventSpeedLU},
    temperature = {type_ = "zeroGradient"},
    mode = "overwrite",
    name = "vent_top",
  })

  vox:addSensor(
  {
    min = {C_L, C_L, mz - 2*C_L},
    max = {mx, my, mz - C_L},
    name = "vent_top_sensor"
  })

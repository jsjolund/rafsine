package.path = package.path .. "./?.lua;lua/?.lua"
require "problems/jet_chamber/settings"
require "VoxelGeometry"

print("Time-step : " .. uc:N_to_s(1) .. " s")
print("Creating geometry of size ", nx, ny, nz)
vox = VoxelGeometry(nx, ny, nz)

-- Length of a voxel in meters
C_L = uc:C_L()
print("C_L = "..C_L.." m")

-- Set domain boundary conditions
vox:addWallXmin()
-- vox:addWallXmax()
-- vox:addWallYmin()
-- vox:addWallYmax()
-- vox:addWallZmin()
-- vox:addWallZmax()

ventSpeed = 0.1
ventSpeedLU = uc:ms_to_lu(ventSpeed)
ventSize = 1

-- expected_flow = ventSpeed * (mx - 2*C_L) * (my - 2*C_L)
-- print("Expected flow = "..expected_flow.." m3/s")

-- Set an inlet on one wall
vox:addQuadBC(
  {
    origin = {0, my/2-ventSize/2, mz/2-ventSize/2},
    dir1 = {0, ventSize, 0},
    dir2 = {0, 0, ventSize},
    typeBC = "inlet",
    normal = {1, 0, 0},
    velocity = {ventSpeedLU, 0, 0},
    temperature = {
      type_ = "constant",
      value = 10
    },
    mode = "overwrite",
    name = "input",
  })

vox:addSensor(
  {
    min = {C_L*2, my/2-ventSize/2, mz/2-ventSize/2},
    max = {C_L*2, my/2+ventSize/2, mz/2+ventSize/2},
    name = "input_sensor"
  })

--Set an outlet on another wall
vox:addQuadBC(
  {
    origin = {mx, 0, 0},
    dir1 = {0, my, 0},
    dir2 = {0, 0, mz},
    typeBC = "inlet",
    normal = {-1, 0, 0},
    velocity = {ventSpeedLU, 0, 0},
    temperature = {type_ = "zeroGradient"},
    mode = "overwrite",
    name = "output",
  })

-- vox:addSensor(
--   {
--     min = {mx-C_L*2, C_L, C_L},
--     max = {mx-C_L*2, my-C_L*2, mz-C_L*2},
--     name = "output_sensor"
--   })

-- boxSize = 1.0
-- vox:addSolidBox(
--   {
--     name = "box",
--     min = {
--       mx/2 - boxSize/2,
--       my/2 - boxSize/2,
--       mz/2 - boxSize/2
--     },
--     max = {
--       mx/2 + boxSize/2,
--       my/2 + boxSize/2,
--       mz/2 + boxSize/2
--     },
--     temperature = 20,
--   })

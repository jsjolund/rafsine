package.path = package.path .. "./?.lua;lua/?.lua"
require "problems/simple/settings"
require "VoxelGeometry"

print("Time-step : " .. uc:N_to_s(1) .. " s")
print("Creating geometry of size ", nx, ny, nz)
vox = VoxelGeometry(nx, ny, nz)

-- Length of a voxel in meters
C_L = uc:C_L()
print("C_L : "..C_L.." m")

-- Set domain boundary conditions
vox:addWallXmin()
vox:addWallXmax()
vox:addWallYmin()
vox:addWallYmax()
vox:addWallZmin()
vox:addWallZmax()

ventSpeed = uc:ms_to_lu(0.5)

-- Set an inlet on one wall
vox:addQuadBC(
  {
    origin = {C_L, C_L, 0},
    dir1 = {mx - 2*C_L, 0, 0},
    dir2 = {0, my - 2*C_L, 0},
    typeBC = "inlet",
    normal = {0, 0, 1},
    velocity = {0, 0, ventSpeed},
    temperature = {
      type_ = "constant",
      value = 10
    },
    mode = "overwrite",
    name = "vent bottom",
  })

--Set an outlet on another wall
vox:addQuadBC(
  {
    origin = {C_L, C_L, mz},
    dir1 = {mx - 2*C_L, 0, 0},
    dir2 = {0, my - 2*C_L, 0},
    typeBC = "inlet",
    normal = {0, 0, -1},
    velocity = {0, 0, ventSpeed},
    temperature = {type_ = "zeroGradient"},
    mode = "overwrite",
    name = "vent top",
  })

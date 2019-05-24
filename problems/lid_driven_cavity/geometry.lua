package.path = package.path .. "./?.lua;lua/?.lua"
require "problems/lid_driven_cavity/settings"
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

lidSpeed = 1.0
lidSpeedLU = uc:ms_to_lu(lidSpeed)

vox:addQuadBC(
  {
    origin = {C_L, C_L, mz},
    dir1 = {mx - 2*C_L, 0, 0},
    dir2 = {0, my - 2*C_L, 0},
    typeBC = "inlet",
    normal = {1, 0, -1},
    velocity = {lidSpeedLU, 0, 0},
    temperature = {
      type_ = "constant",
      value = 20
    },
    mode = "overwrite",
    name = "lid",
  })


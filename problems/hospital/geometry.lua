package.path = package.path .. "./?.lua;lua/?.lua"
require "problems/hospital/settings"
require "VoxelGeometry"

print("Time-step : " .. uc:N_to_s(1) .. " s")
print("Creating geometry of size ", nx, ny, nz)

vox = VoxelGeometry(nx, ny, nz)

-- Set domain boundary conditions
vox:addWallXmin()
vox:addWallXmax()
vox:addWallYmin()
vox:addWallYmax()
vox:addWallZmin()
vox:addWallZmax()

ventilationSpeed = uc:ms_to_lu(1)

-- Set an inlet on one wall
vox:addQuadBC(
{
  origin = {0.0, 3.6, 1.0},
  dir1 = {0.0, 0.5, 0.0},
  dir2 = {0.0, 0.0, 0.2},
  typeBC = "inlet",
  normal = {1, 0, 0},
  velocity = {ventilationSpeed, 0.0, 0.0},
  temperature = {
    type_ = "constant",
    value = 10.0
  },
  mode = "overwrite",
  name = "inlet",
})

--Set an outlet on another wall
vox:addQuadBC(
{
  origin = {7.2, 1.0, 1.0},
  dir1 = {0.0, 0.5, 0.0},
  dir2 = {0.0, 0.0, 0.2},
  typeBC = "inlet",
  normal = {-1, 0, 0},
  velocity = {ventilationSpeed, 0.0, 0.0},
  temperature = {type_ = "zeroGradient"},
  mode = "overwrite",
  name = "outlet",
})

-- create a function to add a bed
function addBed(params)
  local width = 0.6
  local lenght = 1.8
  local height = 0.75
  local min = {
    params.center[1] - width / 2,
    params.center[2] - lenght / 2,
    0
  }
  local max = {
    params.center[1] + width / 2,
    params.center[2] + lenght / 2,
    height
  }
  -- create the bed itself
  vox:addSolidBox({min = min, max = max})
  -- add a hot source on top
  vox:addSolidBox(
  {
    name = "patient"..params.id,
    min = {
      params.center[1] - 0.5 * width / 2,
      params.center[2] - 0.8 * lenght / 2,
      height
    },
    max = {
      params.center[1] + 0.5 * width / 2,
      params.center[2] + 0.8 * lenght / 2,
      height + 0.2
    },
    temperature = 37,
  })
end

-- add four beds
addBed({center = {2, 0.9}, id = 1})
addBed({center = {2, 6.3}, id = 2})
addBed({center = {5.2, 0.9}, id = 3})
addBed({center = {5.2, 6.3}, id = 4})

-- add a doctor in the center
vox:addSolidBox(
{
  name = "doctor",
  min = {
    3.6,
    3.6,
    0
  },
  max = {
    3.6 + 0.6,
    3.6 + 0.3,
    1.8
  },
  temperature = 37,
})

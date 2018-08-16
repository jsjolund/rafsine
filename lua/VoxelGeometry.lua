-- uses the penlight library for its class
require "pl"
utils.import "pl.class"
require "operators"
require "helpers"

VoxelGeometry = class()
function VoxelGeometry:_init(nx, ny, nz)
  self.nx = nx
  self.ny = ny
  self.nz = nz
  self.VOX_EMPTY = -1
  self.VOX_FLUID = 0
  self.voxGeoAdapter = 0
end

-- Add walls on the domain boundaries
function VoxelGeometry:addWallXmin()
  voxGeoAdapter:addWallXmin()
end
function VoxelGeometry:addWallYmin()
  voxGeoAdapter:addWallYmin()
end
function VoxelGeometry:addWallZmin()
  voxGeoAdapter:addWallZmin()
end
function VoxelGeometry:addWallXmax()
  voxGeoAdapter:addWallXmax()
end
function VoxelGeometry:addWallYmax()
  voxGeoAdapter:addWallYmax()
end
function VoxelGeometry:addWallZmax()
  voxGeoAdapter:addWallZmax()
end

function VoxelGeometry:addQuadBC(params)
  temperatureRelPos = 0
  temperatureValue = 0/0
  temperatureType = "none"
  if (params.temperature) then
    if (params.temperature.value) then
      temperatureValue = params.temperature.value
    end
    if (params.temperature.type_) then
      temperatureType = params.temperature.type_
    end
    if (params.temperature.rel_pos) then
      temperatureRelPos = params.temperature.rel_pos
    end
  end
  voxGeoAdapter:addQuadBC(
    params.name,
    params.mode,
    params.origin[1], params.origin[2], params.origin[3],
    params.dir1[1], params.dir1[2], params.dir1[3],
    params.dir2[1] ,params.dir2[2], params.dir2[3],
    params.normal[1], params.normal[2], params.normal[3],
    params.typeBC,
    temperatureType,
    temperatureValue,
    params.velocity[1], params.velocity[2], params.velocity[3],
    temperatureRelPos
  )
end

function VoxelGeometry:addSolidBox(params)
  temperatureValue = 0/0
  if (params.temperature) then
    temperatureValue = params.temperature
  end
  voxGeoAdapter:addSolidBox(
    params.name,
    params.min[1],params.min[2],params.min[3],
    params.max[1],params.max[2],params.max[3],
    temperatureValue
  )
end

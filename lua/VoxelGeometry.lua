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
  -- self.voxGeoAdapter = 0
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

function VoxelGeometry:addSensor(params)
  local name = ""
  if (params.name) then
    name = params.name
  end
  voxGeoAdapter:addSensor(
    name,
    params.min[1], params.min[2], params.min[3],
    params.max[1], params.max[2], params.max[3]
  )
end

function VoxelGeometry:addQuadBC(params)
  local temperatureRelPos = 0/0
  local temperatureValue = 0/0
  local temperatureType = "none"
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

  local velocityX = 0/0
  local velocityY = 0/0
  local velocityZ = 0/0
  if (params.velocity) then 
    velocityX = params.velocity[1]
    velocityY = params.velocity[2]
    velocityZ = params.velocity[3]
  end

  local name = ""
  if (params.name) then
    name = params.name
  end

  voxGeoAdapter:addQuadBC(
    name,
    params.mode,
    params.origin[1], params.origin[2], params.origin[3],
    params.dir1[1], params.dir1[2], params.dir1[3],
    params.dir2[1] ,params.dir2[2], params.dir2[3],
    params.normal[1], params.normal[2], params.normal[3],
    params.typeBC,
    temperatureType,
    temperatureValue,
    velocityX,
    velocityY,
    velocityZ,
    temperatureRelPos
  )
end

function VoxelGeometry:addSolidBox(params)
  local temperatureValue = 0/0
  if (params.temperature) then
    temperatureValue = params.temperature
  end
  local name = ""
  if (params.name) then
    name = params.name
  end
  voxGeoAdapter:addSolidBox(
    name,
    params.min[1],params.min[2],params.min[3],
    params.max[1],params.max[2],params.max[3],
    temperatureValue
  )
end

function VoxelGeometry:makeHollow(params)
  faceMinX = false
  faceMinY = false
  faceMinZ = false
  faceMaxX = false
  faceMaxY = false
  faceMaxZ = false
  if (params.faces) then
    faceMinX = params.faces.xmin ~= nil
    faceMinY = params.faces.ymin ~= nil
    faceMinZ = params.faces.zmin ~= nil
    faceMaxX = params.faces.xmax ~= nil
    faceMaxY = params.faces.ymax ~= nil
    faceMaxZ = params.faces.zmax ~= nil
  end
  voxGeoAdapter:makeHollow(
    params.min[1],
    params.min[2],
    params.min[3],
    params.max[1],
    params.max[2],
    params.max[3],
    faceMinX,
    faceMinY,
    faceMinZ,
    faceMaxX,
    faceMaxY,
    faceMaxZ
  )
end
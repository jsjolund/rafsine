require "VoxelGeometry"
require "settings"

--[[
voxGeo = VoxelGeometry(1,1,1)
voxGeo:loadFromFile("../geo/geometry.vox")

faceLeftServerTypes = voxGeo:extractPlane({
  origin = {1.8,0.6,0.0},
  dir1   = {0.0,3.0,0.0},
  dir2   = {0.0,0.0,2.0},
  noborder = true,
})

-- write the extraction to a file
local f = io.open("faceLeft.txt", "w")
for i,row in ipairs(faceLeftServerTypes) do
  for j,val in ipairs(row) do
    f:write(string.format("%4i",val) )
  end
  f:write("\n")
end
--]]
--[[
voxTemp = VoxelGeometry(1,1,1)
voxTemp:loadFromFile("../output/averageTemperature.vox")


faceLeftServerTemperatures = voxTemp:extractPlane({
  origin = {1.8,0.6,0.0},
  dir1   = {0.0,3.0,0.0},
  dir2   = {0.0,0.0,2.0},
  --noborder = true,
})

faceRightServerTemperatures = voxTemp:extractPlane({
  origin = {3.0,0.6,0.0},
  dir1   = {0.0,3.0,0.0},
  dir2   = {0.0,0.0,2.0},
})

-- write the extraction to a file
-- matrix data
local f = io.open("subsetData/faceLeftTemperatureMatrix.txt", "w")
for i,row in ipairs(faceLeftServerTemperatures) do
  for j,data in ipairs(row) do
    f:write(string.format("%2.2f ",data.value) )
  end
  f:write("\n")
end

--list data
local f = io.open("subsetData/faceLeftTemperatureList.txt", "w")
f:write(string.format("#%4s %5s %5s %5s\n","x","y","z","temperature"))
for i,row in ipairs(faceLeftServerTemperatures) do
  for j,data in ipairs(row) do
    local p = data.position
    f:write(string.format("%5f %5f %5f %2.2f\n",p[1],p[2],p[3],data.value) )
  end
  f:write("\n")
end


local f = io.open("subsetData/faceRightTemperatureMatrix.txt", "w")
for i,row in ipairs(faceRightServerTemperatures) do
  for j,data in ipairs(row) do
    f:write(string.format("%2.2f ",data.value) )
  end
  f:write("\n")
end

--list data
local f = io.open("subsetData/faceRightTemperatureList.txt", "w")
f:write(string.format("#%4s %5s %5s %5s\n","x","y","z","temperature"))
for i,row in ipairs(faceRightServerTemperatures) do
  for j,data in ipairs(row) do
    local p = data.position
    f:write(string.format("%5f %5f %5f %2.2f\n",p[1],p[2],p[3],data.value) )
  end
  f:write("\n")
end
--]]
voxVx = VoxelGeometry(1, 1, 1)
voxVy = VoxelGeometry(1, 1, 1)
voxVz = VoxelGeometry(1, 1, 1)
voxVx:loadFromFile("../output/averageVelocityX.vox")
voxVy:loadFromFile("../output/averageVelocityY.vox")
voxVz:loadFromFile("../output/averageVelocityZ.vox")

--compute the norm
nx = voxVx.nx
ny = voxVx.ny
nz = voxVx.nz
voxVnorm = VoxelGeometry(nx, ny, nz)
for i = 1, nx do
  for j = 1, ny do
    for k = 1, nz do
      local vx = voxVx.data[i][j][k]
      local vy = voxVy.data[i][j][k]
      local vz = voxVz.data[i][j][k]
      voxVnorm.data[i][j][k] = math.sqrt(vx * vx + vy * vy + vz * vz)
    end
  end
end

--extract a slice through the center of the room
sliceVx =
  voxVx:extractPlane(
  {
    origin = {2.4, 0.0, 0.0},
    dir1 = {0.0, 6.0, 0.0},
    dir2 = {0.0, 0.0, 2.8}
  }
)
sliceVy =
  voxVy:extractPlane(
  {
    origin = {2.4, 0.0, 0.0},
    dir1 = {0.0, 6.0, 0.0},
    dir2 = {0.0, 0.0, 2.8}
  }
)
sliceVz =
  voxVz:extractPlane(
  {
    origin = {2.4, 0.0, 0.0},
    dir1 = {0.0, 6.0, 0.0},
    dir2 = {0.0, 0.0, 2.8}
  }
)
sliceVnorm =
  voxVnorm:extractPlane(
  {
    origin = {2.4, 0.0, 0.0},
    dir1 = {0.0, 6.0, 0.0},
    dir2 = {0.0, 0.0, 2.8}
  }
)

-- write the extraction to a file
-- matrix data
local f = io.open("subsetData/sliceVnormMatrix.txt", "w")
for i, row in ipairs(sliceVnorm) do
  for j, data in ipairs(row) do
    f:write(string.format("%1.3e ", data.value))
  end
  f:write("\n")
end

--list data
local f = io.open("subsetData/sliceVList.txt", "w")
f:write(string.format("#%4s %5s %5s %9s %9s %9s %8s\n", "x", "y", "z", "Vx", "Vy", "Vz", "Vnorm"))
for i, row in ipairs(sliceVnorm) do
  for j, data in ipairs(row) do
    local p = data.position
    local vx = sliceVx[i][j].value
    local vy = sliceVy[i][j].value
    local vz = sliceVz[i][j].value
    f:write(string.format("%1.3f %1.3f %1.3f % 1.3e % 1.3e % 1.3e %1.3e\n", p[1], p[2], p[3], vx, vy, vz, data.value))
  end
  f:write("\n")
end

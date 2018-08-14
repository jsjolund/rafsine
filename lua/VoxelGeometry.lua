-- uses the penlight library for its class
require "pl"
utils.import "pl.class"
require "operators"
require "helpers"

local mk = require("multikey")

VoxelGeometry = class()
function VoxelGeometry:_init(nx, ny, nz)
  self.nx = nx
  self.ny = ny
  self.nz = nz
  self.VOX_EMPTY = -1
  self.VOX_FLUID = 0
  self.data = {}
  for i = 1, nx do
    self.data[i] = {} -- create a new row
    for j = 1, ny do
      self.data[i][j] = {}
      for k = 1, nz do
        self.data[i][j][k] = self.VOX_FLUID
      end
    end
  end
  -- next available voxe type
  self.newtype = 1
  -- multikey table to store the type for each description
  self.voxtype = {}
  -- store the description for each type
  self.voxdetail = {}
  --table.insert(self.voxtypes, {description='empty'} )
end

-- function to set a voxel based on a 3D position {x,y,z}
function VoxelGeometry:set(p, value)
  self.data[p[1]][p[2]][p[3]] = value
end
function VoxelGeometry:get(p)
  --print("p=",expand(p))
  return self.data[p[1]][p[2]][p[3]]
end

function VoxelGeometry:saveToFile(fileName)
  print("fileName : ", fileName)
  f = io.open(fileName, "w")
  f:write(self.nx .. " " .. self.ny .. " " .. self.nz .. "\n")
  for k = 1, self.nz do
    for j = 1, self.ny do
      for i = 1, self.nx do
        f:write(self.data[i][j][k])
        f:write(" ")
      end
      f:write("\n")
    end
    f:write("\n")
  end
end

function VoxelGeometry:loadFromFile(fileName)
  local f = assert(io.open(fileName, "r"))
  local line_nbr = -1
  for line in f:lines() do
    if line ~= "" then --skip empty lines
      if line_nbr == -1 then -- initialise the array to the right size
        self.nx, self.ny, self.nz = utils.splitv(line, " ")
        self.data = {}
        for i = 1, self.nx do
          self.data[i] = {} -- create a new row
          for j = 1, self.ny do
            self.data[i][j] = {}
            for k = 1, self.nz do
              self.data[i][j][k] = 0
            end
          end
        end
      else
        for i, val in ipairs(utils.split(line)) do
          --print("line="..line_nbr, string.format("%5i %5i %5i   %5i", i, 1 + line_nbr%self.ny, 1 + math.floor(line_nbr/self.ny), val))
          self.data[i][1 + line_nbr % self.ny][1 + math.floor(line_nbr / self.ny)] = val
        end
      end
      line_nbr = line_nbr + 1
    end
  end
end

--function to get the type from the description
function VoxelGeometry:getType(description)
  if description.typeBC == "fluid" then
    return self.VOX_FLUID
  end
  if description.typeBC == "empty" then
    return self.VOX_EMPTY
  end
  local n = description.normal
  if (description.velocity) then
    local v = description.velocity
    if (description.temperature) then
      local t = description.temperature
      mk.get(self.voxtype, description.typeBC, n[1], n[2], n[3], v[1], v[2], v[3], t.type_, t.value, t.rel_pos, type_id)
    else
      return mk.get(self.voxtype, description.typeBC, n[1], n[2], n[3], v[1], v[2], v[3])
    end
  else
    return mk.get(self.voxtype, description.typeBC, n[1], n[2], n[3])
  end
end

-- function to set the type from a description
function VoxelGeometry:setType(description, type_id)
  local n = description.normal
  if (description.velocity) then
    local v = description.velocity
    if (description.temperature) then
      local t = description.temperature
      mk.put(self.voxtype, description.typeBC, n[1], n[2], n[3], v[1], v[2], v[3], t.type_, t.value, type_id)
    else
      mk.put(self.voxtype, description.typeBC, n[1], n[2], n[3], v[1], v[2], v[3], type_id)
    end
  else
    mk.put(self.voxtype, description.typeBC, n[1], n[2], n[3], type_id)
  end
end

-- generate a new type of voxel
-- double link voxel type and description
function VoxelGeometry:createNewVoxelType(description)
  local newid = self.newtype
  -- if this type of BC hasn't appeared yet, create a new one
  if self:getType(description) == nil then
    --attach type to description
    self:setType(description, newid)
    --attach desription to type
    self.voxdetail[newid] = description
  else
    error("Type already exist:", self:getType(desription))
  end
  -- increment the next available type
  self.newtype = self.newtype + 1
  return newid
end

-- return the correct voxel type for the boundary
-- create a new one if the boundary does not exist already
function VoxelGeometry:getBCVoxelType(params)
  -- if the parameters correspond to a type, then use it
  if self:getType(params) then
    return self:getType(params)
  else -- otherwise, create a new type based on the parameters
    return self:createNewVoxelType(params)
  end
end

-- function to compute a new type for intersection of two types
-- or use one already existing
function VoxelGeometry:getBCIntersectType(position, params)
  -- type of the existing voxel
  local vox1 = self:get(position)

  -- normal of the exiting voxel
  local n1 = vector(self.voxdetail[vox1].normal)
  -- normal of the new boundary
  local n2 = vector(params.normal)
  -- build a new vector, sum of the two vectors
  local n = n1 + n2
  -- if the boundaries are opposite, they cannot be compatible, so otherwrite with the new boundary
  if n1 == -n2 then
    n = n2
  end

  -- name of the existing boundary
  local name1 = self.voxdetail[vox1].name
  -- name of the new boundary
  local name2 = params.name

  --TODO this suppose they have the same boundary type
  return self:getBCVoxelType(
    {
      typeBC = params.typeBC,
      normal = n,
      name = name1 .. ", " .. name2
    }
  )
end

-- General function to add boundary conditions on a quad
-- Unit are in nodes
-- Parameters:
--   origin       origin of the quad
--   dir1         1st quad edge
--   dir2         2nd quad edge
--   typeBC       type of boundary condition
--   normal       vector normal to the BC
--   mode         describe how the BC is added to the domain
--                'overwrite' replace whatever boundary condition that was there before
--                'intersect' create new boundary types on intersection
--                'fill'      only add boundary if there isn't one already
-- Optional parameters:
--   velocity     velocity at the BC
--   temperature  temperature at the BC
function VoxelGeometry:addQuadBC_node_units(params)
  -- get the type of voxel based on the params or create a new one
  local voxtype = self:getBCVoxelType(params)
  local l1 = norm(vector(params.dir1))
  local l2 = norm(vector(params.dir2))
  local dir1 = (1.0 / l1) * vector(params.dir1)
  local dir2 = (1.0 / l2) * vector(params.dir2)
  for i = 0, l1 do
    for j = 0, l2 do
      local p = vector(params.origin) + i * dir1 + j * dir2
      -- there is a boundary already
      if self:get(p) ~= self.VOX_EMPTY and self:get(p) ~= self.VOX_FLUID then
        if params.mode == "overwrite" then -- over write whatever type was there
          self:set(p, voxtype)
        elseif params.mode == "intersect" then
          -- the boundary is intersecting another boundary
          local t = self:getBCIntersectType(p, params)
          self:set(p, t)
        elseif params.mode == "fill" then
        -- do nothing
        end
      else
        -- replacing empty voxel
        self:set(p, voxtype)
      end
    end
  end
  return voxtype
end

--function to add boundary on a quad
--the quad is defined in real units
function VoxelGeometry:addQuadBC(params)
  -- in lua array starts at 1 (not 0) so everything needs to be shifted
  local origin = vector(uc:m_to_LUA(params.origin))
  -- directions do not need to be shifted
  --local dir1   = uc:m_to_lu(params.dir1)
  --local dir2   = uc:m_to_lu(params.dir2)
  local dir1 = uc:m_to_LUA(vector(params.origin) + vector(params.dir1)) - origin
  local dir2 = uc:m_to_LUA(vector(params.origin) + vector(params.dir2)) - origin
  --print("New quad boundary", params.name)
  --print("origin", expand(origin))
  --print("dir1",   expand(dir1))
  --print("dir2",   expand(dir2))
  --print("origin.y + dir1.y = ", uc:m_to_LUA(params.origin[2] + params.dir1[2]))
  --print("origin.y + dir1.y = ", uc:m_to_LUA(params.origin[2]) + uc:m_to_lu(params.dir1[2]))
  --TODO convert velocity
  local velocity = params.velocity
  -- TODO convert temperature
  local temperature = params.temperature
  return self:addQuadBC_node_units(
    {
      origin = origin,
      dir1 = dir1,
      dir2 = dir2,
      typeBC = params.typeBC,
      normal = params.normal,
      mode = params.mode,
      name = params.name,
      velocity = velocity,
      temperature = temperature
    }
  )
end

-- Add walls on the domain boundaries
function VoxelGeometry:addWallXmin()
  -- normal to the face
  local n = {1, 0, 0}
  self:addQuadBC_node_units(
    {
      origin = {1, 1, 1},
      dir1 = {0, self.ny - 1, 0},
      dir2 = {0, 0, self.nz - 1},
      typeBC = "wall",
      normal = n,
      mode = "intersect",
      name = "xmin"
    }
  )
end

-- Add walls on the domain boundaries
function VoxelGeometry:addWallXmax()
  -- normal to the face
  local n = {-1, 0, 0}
  self:addQuadBC_node_units(
    {
      origin = {self.nx, 1, 1},
      dir1 = {0, self.ny - 1, 0},
      dir2 = {0, 0, self.nz - 1},
      typeBC = "wall",
      normal = n,
      mode = "intersect",
      name = "xmax"
    }
  )
end

-- Add walls on the domain boundaries
function VoxelGeometry:addWallYmin()
  -- normal to the face
  local n = {0, 1, 0}
  self:addQuadBC_node_units(
    {
      origin = {1, 1, 1},
      dir1 = {self.nx - 1, 0, 0},
      dir2 = {0, 0, self.nz - 1},
      typeBC = "wall",
      normal = n,
      mode = "intersect",
      name = "ymin"
    }
  )
end

-- Add walls on the domain boundaries
function VoxelGeometry:addWallYmax()
  -- normal to the face
  local n = {0, -1, 0}
  self:addQuadBC_node_units(
    {
      origin = {1, self.ny, 1},
      dir1 = {self.nx - 1, 0, 0},
      dir2 = {0, 0, self.nz - 1},
      typeBC = "wall",
      normal = n,
      mode = "intersect",
      name = "ymax"
    }
  )
end

function VoxelGeometry:addWallZmin()
  -- normal to the face
  local n = {0, 0, 1}
  self:addQuadBC_node_units(
    {
      origin = {1, 1, 1},
      dir1 = {self.nx - 1, 0, 0},
      dir2 = {0, self.ny - 1, 0},
      typeBC = "wall",
      normal = n,
      mode = "intersect",
      name = "zmin"
    }
  )
end

function VoxelGeometry:addWallZmax()
  -- normal to the face
  local n = {0, 0, -1}
  self:addQuadBC_node_units(
    {
      origin = {1, 1, self.nz},
      dir1 = {self.nx - 1, 0, 0},
      dir2 = {0, self.ny - 1, 0},
      typeBC = "wall",
      normal = n,
      mode = "intersect",
      name = "zmax"
    }
  )
end

-- function to remove the inside of a box
-- can also remove faces with the option 'faces' parameter
function VoxelGeometry:makeHollow(params)
  -- convert to LUA node units
  local min = uc:m_to_LUA(params.min) + vector({1, 1, 1})
  local max = uc:m_to_LUA(params.max) - vector({1, 1, 1})
  -- if faces are to be removed then modify min and max
  if params.faces then
    if params.faces.xmin then
      min[1] = min[1] - 1
    end
    if params.faces.ymin then
      min[2] = min[2] - 1
    end
    if params.faces.zmin then
      min[3] = min[3] - 1
    end
    if params.faces.xmax then
      max[1] = max[1] + 1
    end
    if params.faces.ymax then
      max[2] = max[2] + 1
    end
    if params.faces.zmax then
      max[3] = max[3] + 1
    end
  end
  -- remove the inside of the box
  for i = min[1], max[1] do
    for j = min[2], max[2] do
      for k = min[3], max[3] do
        self.data[i][j][k] = self.VOX_EMPTY
      end
    end
  end
end

-- function to add a solid box in the domain
function VoxelGeometry:addSolidBox(params)
  local V, BC
  if params.temperature then
    V = {0, 0, 0}
    BC = "inlet"
    TBC = {
      type_ = "constant",
      value = params.temperature
    }
  else
    V = nil
    BC = "wall"
    TBC = nil
  end

  -- convert to LUA node units
  local min = uc:m_to_LUA(params.min)
  local max = uc:m_to_LUA(params.max)
  self:addQuadBC_node_units(
    {
      origin = min,
      dir1 = {max[1] - min[1], 0, 0},
      dir2 = {0, max[2] - min[2], 0},
      typeBC = BC,
      velocity = V,
      normal = {0, 0, -1},
      temperature = TBC,
      mode = "overwrite",
      name = params.name .. " (bottom)"
    }
  )
  self:addQuadBC_node_units(
    {
      origin = {min[1], min[2], max[3]},
      dir1 = {max[1] - min[1], 0, 0},
      dir2 = {0, max[2] - min[2], 0},
      typeBC = BC,
      velocity = V,
      normal = {0, 0, 1},
      temperature = TBC,
      mode = "intersect",
      name = params.name .. " (top)"
    }
  )
  self:addQuadBC_node_units(
    {
      origin = min,
      dir1 = {0, max[2] - min[2], 0},
      dir2 = {0, 0, max[3] - min[3]},
      typeBC = BC,
      velocity = V,
      normal = {-1, 0, 0},
      temperature = TBC,
      mode = "intersect",
      name = params.name .. " (side x minus)"
    }
  )
  self:addQuadBC_node_units(
    {
      origin = {max[1], min[2], min[3]},
      dir1 = {0, max[2] - min[2], 0},
      dir2 = {0, 0, max[3] - min[3]},
      typeBC = BC,
      velocity = V,
      normal = {1, 0, 0},
      temperature = TBC,
      mode = "intersect",
      name = params.name .. " (side x plus)"
    }
  )
  self:addQuadBC_node_units(
    {
      origin = min,
      dir1 = {max[1] - min[1], 0, 0},
      dir2 = {0, 0, max[3] - min[3]},
      typeBC = BC,
      velocity = V,
      normal = {0, -1, 0},
      temperature = TBC,
      mode = "intersect",
      name = params.name .. " (side y minus)"
    }
  )
  self:addQuadBC_node_units(
    {
      origin = {min[1], max[2], min[3]},
      dir1 = {max[1] - min[1], 0, 0},
      dir2 = {0, 0, max[3] - min[3]},
      typeBC = BC,
      velocity = V,
      normal = {0, 1, 0},
      temperature = TBC,
      mode = "intersect",
      name = params.name .. " (side y plus)"
    }
  )
  --self:makeHollow(params)
  self:makeHollow(
    {
      min = params.min,
      max = params.max,
      faces = {
        xmin = min[1] <= 1,
        ymin = min[2] <= 1,
        zmin = min[3] <= 1,
        xmax = max[1] >= self.nx,
        ymax = max[2] >= self.ny,
        zmax = max[3] >= self.nz
      }
    }
  )
end

require "helpers"
require "NodeDescriptor"
require "NodeD3Q6"
require "FileWriter"
function VoxelGeometry:generateKernel(fileName)
  -- fluid node
  local node = D3Q19Descriptor
  -- temperature node
  local nodeT = D3Q6Descriptor
  local f = FileWriter(fileName)
  local cases = {}
  local statements = {}
  local comments = {}

  -- generate boundary condition depending on voxel type
  for bc_id, bc in ipairs(self.voxdetail) do
    bc.id = bc_id

    local code = {}

    if bc.typeBC == "wall" then
      -- generate half-way boundary condition
      local n = vector(bc.normal)
      -- BC for velocity dfs
      for i, ei in ipairs(node.directions) do
        if ei .. n > 0 then
          local j = node:getOppositeDirection(ei)
          local fi = node:genDistributionName(i)
          table.insert(code, fi .. " = df3D(" .. j - 1 .. ", x,y,z,nx,ny,nz)")
        end
      end
      -- BC for temperature dfs
      for i, ei in ipairs(nodeT.directions) do
        if ei .. n > 0 then
          local j = nodeT:getOppositeDirection(ei)
          table.insert(code, "T" .. i .. " = Tdf3D(" .. j .. ", x,y,z,nx,ny,nz)")
        end
      end
    elseif bc.typeBC == "free_slip" then
      -- generate half-way boundary condition
      local n = vector(bc.normal)
      -- BC for velocity dfs
      for i, ei in ipairs(node.directions) do
        if ei .. n > 0 then
          local j = node:getReflectedDirection(ei, n)
          local fi = node:genDistributionName(i)
          table.insert(code, fi .. " = df3D(" .. j - 1 .. ", x,y,z,nx,ny,nz)")
        end
      end
      -- BC for temperature dfs
      for i, ei in ipairs(nodeT.directions) do
        if ei .. n > 0 then
          local j = nodeT:getReflectedDirection(ei, n)
          table.insert(code, "T" .. i .. " = Tdf3D(" .. j .. ", x,y,z,nx,ny,nz)")
        end
      end
    elseif bc.typeBC == "inlet" then
      -- generate inlet boundary condition
      local v = vector(bc.velocity)
      local n = vector(bc.normal)
      -- BC for velocity dfs
      for i, ei in ipairs(node.directions) do
        if ei .. n > 0 then
          local fi = node:genDistributionName(i)
          local wi = node:getWeight(i)
          local rho = 1
          -- if the velocity is zero, use half-way bounceback instead
          if norm(v) == 0 then
            --print("inlet velocity norm is zero, using bounce-back")
            local j = node:getOppositeDirection(ei)
            table.insert(code, fi .. " = df3D(" .. j - 1 .. ", x,y,z,nx,ny,nz)")
          else
            table.insert(
              code,
              fi .. " = real(" .. wi * rho * (1 + 3 * (ei .. v) + 4.5 * (ei .. v) ^ 2 - 1.5 * (v .. v)) .. ")"
            )
          end
        end
      end
      -- BC for temperature dfs
      local t = vector(bc.temperature)

      for i, ei in ipairs(nodeT.directions) do
        local wi = nodeT:getWeight(i)
        local X = vector({"x", "y", "z"})
        local nb1 = X + n
        if ei .. n > 0 then
          if t.type_ == "constant" then
            table.insert(code, "T" .. i .. " = real(" .. wi * t.value * (1 + 3 * (ei .. v)) .. ")")
          elseif t.type_ == "zeroGradient" then
            -- approximate a first order expansion
            table.insert(code, "T" .. i .. " = Tdf3D(" .. i .. "," .. expand(nb1) .. ",nx,ny,nz)")
          elseif t.type_ == "relative" then
            -- compute the relative node position
            local rel_pos = X + -(1 + uc:m_to_lu(t.rel_pos)) * n
            -- compute macroscopic temperature at the relative position
            local subcode = "real Trel ="
            for i = 1, 6 do
              subcode = subcode .. " + Tdf3D(" .. i .. "," .. expand(rel_pos) .. ",nx,ny,nz)"
            end
            table.insert(code, subcode)
            --print("BC:"..bc_id, expand(rel_pos))
            --table.insert(code, "printf(\" BC:%d, rel BC:%d \\n\", "..bc_id..", voxels[I3D("..expand(rel_pos)..",nx,ny,nz)])")
            local Trel = "(Trel+" .. t.value .. ")"
            table.insert(code, "T" .. i .. " = real(" .. Trel .. "*" .. wi * (1 + 3 * (ei .. v)) .. ")")
          end
        end
      end
    end

    ---[[ -- use if {} blocks
    if (self.voxdetail[bc_id].name) then
      f:comment(self.voxdetail[bc_id].name)
    end
    f:conditionalBlock(
      --"voxels[I3D(x,y,z,nx,ny,nz)]=="..bc_id,
      "voxelID ==" .. bc_id,
      code
    )
    --]]
    table.insert(cases, bc_id)
    table.insert(statements, code)
    table.insert(comments, self.voxdetail[bc_id].name)
  end
  --[[ -- use a switch() block
  f:switchBlock({
    expression = "voxels[I3D(x,y,z,nx,ny,nz)]",
    cases = cases,
    statements = statements,
    comments = comments,
  })
  --]]
end

function VoxelGeometry:extractPlane(params)
  plane = {} -- 2D matrix with the same dimension as the plane
  -- convert meters unit into voxel units
  local origin = vector(uc:m_to_LUA(params.origin))
  local dir1 = vector(uc:m_to_LUA(vector(params.origin) + vector(params.dir1)) - origin)
  local dir2 = vector(uc:m_to_LUA(vector(params.origin) + vector(params.dir2)) - origin)
  -- lenght of each direction in number of ndes
  local l1 = norm(dir1)
  local l2 = norm(dir2)
  -- normalise vectors
  local dir1 = (1.0 / l1) * dir1
  local dir2 = (1.0 / l2) * dir2

  if params.noborder then
    for j = 1, l2 - 1 do
      plane[j] = {} -- create a new row
      for i = 1, l1 - 1 do
        -- compute real world position
        local p_real = vector(params.origin) + (i / l1) * vector(params.dir1) + (j / l2) * vector(params.dir2)
        -- compute node position
        local p = origin + i * dir1 + j * dir2
        plane[j][i] = {position = p_real, value = self:get(p)}
      end
    end
  else
    for j = 0, l2 do
      plane[j + 1] = {} -- create a new row
      for i = 0, l1 do
        local p_real = vector(params.origin) + (i / l1) * vector(params.dir1) + (j / l2) * vector(params.dir2)
        local p = origin + i * dir1 + j * dir2
        plane[j + 1][i + 1] = {position = p_real, value = self:get(p)}
      end
    end
  end
  return plane
end

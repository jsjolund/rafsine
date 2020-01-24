require "pl"
utils.import "pl.class"

require "operators"
local matrix = require "matrix"

-- Describe the node use in the simulation
NodeDescriptor = class()
function NodeDescriptor:_init(D, Q, directions, weights)
  -- dimension of the node (2 or 3)
  self.D = D
  -- numbers of microscopic directions of the node
  self.Q = Q
  -- tables storing microscopic directions
  self.directions = directions
  -- allow the use of operators on the directions
  for _, ei in pairs(self.directions) do
    setmetatable(ei, metaTableVectors)
  end
  -- table storing the weights associated with each direction
  self.weights = weights
end

-- Fucntion to apply the chosen precision of the weight values
function NodeDescriptor:applyPrecision(precision)
  -- Apply the chosen precision to the weights
  for i, w in ipairs(self.weights) do
    -- First find the position divide symbol '/'
    local cut = string.find(w, "/")
    if cut then
      local numerator = string.sub(w, 1, cut - 1)
      local denominator = string.sub(w, cut + 1)
      if precision == "float" then
        numerator = numerator .. ".f"
        denominator = denominator .. ".f"
      else
        numerator = numerator .. "."
        denominator = denominator .. "."
      end
      self.weights[i] = numerator .. "/" .. denominator
    end
  end
end

function NodeDescriptor:getDimensions()
  return self.D
end

function NodeDescriptor:getDirections()
  return self.Q
end

function NodeDescriptor:genName()
  return "D" .. self.D .. "Q" .. self.Q
end

-- generate the name of the ith distribution function
function NodeDescriptor:genDistributionName(i)
  return "f" .. tostring(i - 1)
end

-- generate the name of the ith equilibrium distribution function
function NodeDescriptor:genEquilibriumDistributionName(i)
  return self:genDistributionName(i) .. "eq"
end

-- generate all the names in a table
function NodeDescriptor:genDFNamesList()
  names = vector()
  for i = 1, self.Q do
    names[i] = self:genDistributionName(i)
  end
  return names
end
-- generate all distribution functions names as : (f0,f1,...fQ)
function NodeDescriptor:genAllDistributionNames()
  return table.concat(self:genDFNamesList(), ", ")
end

-- generate the distribution function names and store them in a matrix
function NodeDescriptor:genDFNamesMatrix()
  return matrix {self:genDFNamesList()} ^ "T"
end

-- returns the coordinates for the ith direction
function NodeDescriptor:getDirection(i)
  return self.directions[i]
end

-- returns the weight associated the ith direction
function NodeDescriptor:getWeight(i)
  return self.weights[i]
end

-- returns the index of the opposite direction to a vector
function NodeDescriptor:getOppositeDirection(ei)
  -- function can be used with ei being an index
  if type(ei) == "number" then
    ei = self.directions[ei]
  end
  for j, ej in ipairs(self.directions) do
    if ej == (-ei) then
      return j
    end
  end
end

-- return the index of the relfected vector based on a normal vector
function NodeDescriptor:getReflectedDirection(ei, n)
  -- function can be used with ei being an index
  if type(ei) == "number" then
    ei = self.directions[ei]
  end
  ei_reflected = ei - 2 * (ei .. n) * n
  for j, ej in ipairs(self.directions) do
    if ej == ei_reflected then
      return j
    end
  end
end

-- generate a list of pairs of (directions, opposite direction)
function NodeDescriptor:getPairs()
  local zero
  local fipairs = {}
  for i, ei in ipairs(self.directions) do
    if ei .. ei == 0 then
      zero = i
    else
      local j = self:getOppositeDirection(ei)
      --check the pair does not exist already
      if fipairs[j] == nil then
        fipairs[i] = j
      end
    end
  end
  return {
    zero = zero,
    fipairs = fipairs
  }
end

require "NodeDescriptorD2Q9"
require "NodeDescriptorD3Q15"
require "NodeDescriptorD3Q19"
require "NodeDescriptorD3Q27"

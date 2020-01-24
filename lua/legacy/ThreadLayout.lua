require "pl"
utils.import "pl.class"
require "NodeDescriptor"
require "ModelParameters"

--[[
Define the layout of CUDA threads in a grid of blocks and generate the appropriate code.
Parameters:
- layout:
    1D_BLOCK, 2D_BLOCK
- maxThreadsPerBlock: (only works in 2D)
    maximal number of threads contained in a block
    if maxThreadsPerBlock < LatSizeX then the there will be several blocks along that direction
- gridName: name to be used for the grid  size in the generated code
- blockName: ...................... block size .....................
--]]
ThreadLayout = class()
function ThreadLayout:_init(args)
  self.layout = args.layout or "1D_BLOCK"
  --  TODO: if not( self.layout == "1D_BLOCK" or self.layout == "2D_BLOCK" ) then error("Unknown layout type : "..self.layout) end
  self.maxThreadsPerBlock = args.maxThreadsPerBlock or nil
  self.gridName = args.gridName or "gridSize"
  self.blockName = args.blockName or "blockSize"
  -- compute the correct block and grid size
  if node.D == 2 then
    if self.maxThreadsPerBlock then
      self.block = {self.maxThreadsPerBlock, 1, 1}
      --handle cases when sizeX is not a multiple of maxThreadsPerBlock
      local condition = model.sizeX.name .. "%" .. self.maxThreadsPerBlock .. " == 0"
      self.grid = {model.sizeX.name / self.maxThreadsPerBlock + ternary(condition, 0, 1), model.sizeY.name, 1}
      self.needNodePositionChecking = true
    else
      self.block = {model.sizeX.name, 1, 1}
      self.grid = {model.sizeY.name, 1, 1}
    end
  else -- node.D == 3
    self.block = {model.sizeX.name, 1, 1}
    self.grid = {model.sizeY.name, model.sizeZ.name, 1}
  end
  -- store methods to compute node position
  if node.D == 2 then
    if self.maxThreadsPerBlock then
      self.node_idx = {"threadIdx.x" + "blockIdx.x" * self.maxThreadsPerBlock, "blockIdx.y"}
    else
      self.node_idx = {"threadIdx.x", "blockIdx.x"}
    end
  else -- node.D == 3
    self.node_idx = {"threadIdx.x", "blockIdx.x", "blockIdx.y"}
  end
end

function ThreadLayout:genNodeDefList(X)
  return merge({"int ", X, " = ", self.node_idx})
end

-- generate the block size declaration in CUDA
function ThreadLayout:genThreadBlockSize()
  local block = table.concat(self.block, ", ")
  return "dim3 " .. self.blockName .. "(" .. block .. ")"
end

-- generate the grid size declaration in CUDA
function ThreadLayout:genThreadGridSize()
  local grid = table.concat(self.grid, ", ")
  return "dim3 " .. self.gridName .. "(" .. grid .. ")"
end

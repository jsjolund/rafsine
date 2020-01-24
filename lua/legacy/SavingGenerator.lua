-- Generate the code to save the simulation results to files (vtk)
SavingGenerator = class()
function SavingGenerator:_init(args)
  args = args or {}
  --number of time-steps between each call to the update function
  self.updateInterval = args.interval or 100
  --quantities to save (reorganise data for easier use)
  self.quantities = {}
  if args.quantities then
    for _, q in pairs(args.quantities) do
      self.quantities[q] = true
    end
    --ask the main generator to store the quantities that need to be saved
    if self.quantities["velocity"] then
      main.storeVelocity = true
    end
    if self.quantities["density"] then
      main.storeDensity = true
    end
  end
end

function SavingGenerator:genDependencies()
  return "SaveToVTK.hpp"
end

function SavingGenerator:genInit()
  local inits = {}
  -- Allocate CPU memory to store the velocity
  if self.quantities["velocity"] then
    for _, v in pairs(model.velocity_name) do
      table.insert(inits, memory:genMemAllocCPU(v:upper()))
    end
  end
  if self.quantities["density"] then
    table.insert(inits, memory:genMemAllocCPU(model.density_name:upper()))
  end
  return inits
end

function SavingGenerator:genUpdate()
  local code = {}
  -- Copy the quantities stored on the GPU to the CPU
  if self.quantities["velocity"] then
    for _, v in pairs(model.velocity_name) do
      table.insert(code, v:upper() .. " = " .. v:upper() .. "_d")
    end
  end
  if self.quantities["density"] then
    table.insert(code, model.density_name:upper() .. " = " .. model.density_name:upper() .. "_d")
  end
  -- Call the saveToVTK functions
  if self.quantities["velocity"] then
    -- build all the required args
    local args = {}
    table.insert(args, model.time_name)
    for _, s in pairs(model.size_names) do
      table.insert(args, s)
    end
    if node.D == 2 then
      table.insert(args, 1)
    end
    for _, v in pairs(model.velocity_name) do
      table.insert(args, memory:genPointer(v:upper()))
    end
    if node.D == 2 then
      table.insert(args, "NULL")
    end
    --local vz = ( model.D == 2 and "NULL" or model.velocity_name[3]:upper() )
    table.insert(code, "WriteVelocityVTK(" .. table.concat(args, ", ") .. ")")
  --int time, int sizeX, int sizeY, int sizeZ, real* vx, real* vy, real* vz)
  end
  if self.quantities["density"] then
  -- TODO .........
  end
  return code
end

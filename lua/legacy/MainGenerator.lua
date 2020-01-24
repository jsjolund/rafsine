require "FileWriter"
--require "helpers.lua"

-- generate the main file as well as the dependencies
MainGenerator = class()
function MainGenerator:_init(fileName)
  -- name of the main CUDA file
  self.fileName = fileName or "main.cu"
  -- file writter for main.cu
  self.filewriter = FileWriter(src_destination_folder .. "/" .. self.fileName)
  -- short version
  self.f = self.filewriter
  self.w = function(text)
    if type(text) == "string" then
      self.filewriter:writeLine(text)
    elseif type(text) == "table" then
      for _, line in pairs(text) do
        self.w(line)
      end
    end
  end
  self.modules = {}
  -- method used for the initialisation
  self.initMethod = nil
  --TODO: add a store parameter to the moment class to allow to store or not any moment
  -- True if the velocity needs to be stored in memory
  self.storeVelocity = false
  -- True if the density needs to be stored in memory
  self.storeDensity = false
end

-- Add a module generator to the system
-- Module are code generators that add a few more codes to the program
-- Examples of modules are:
--   RenderGenerator    add real-time OpenGL rendering to the program
--   SavingGenerator    add the ability to save the simulation results
--   ...
function MainGenerator:addModule(module_)
  -- add the module to the list of module
  table.insert(self.modules, module_)
  -- Initialise the module if it needs to
  -- TODO: unused?
  if module_.setUp then
    module_:setUp()
  end
end

function MainGenerator:setInitialisationMethod(method)
  self.initMethod = method
end

function MainGenerator:generate()
  self:genIntroduction()
  self.f:skipLine()
  self.f:comment("Define the precision use for describing real number")
  self.w("typedef " .. model.precision .. " real")
  self.f:genDependencies()
  self.f:include("helper_cuda.h")
  self.f:include("DF_array_GPU.h")
  for _, m in pairs(self.modules) do
    if m.genDependencies then
      self.f:include(m:genDependencies())
    end
  end
  self.f:include(kernel.fileName)
  self.f:skipLine()
  self:genMain()
end

-- generate an introduction to main file
function MainGenerator:genIntroduction()
  self.f:startComment()
  self.w("\\file")
  self.w("    " .. self.fileName)
  self.w("\\brief")
  self.w("    Lattice Boltzmann Simulation on the GPU, using a " .. node:genName())
  self.w("    This code was generated at " .. currentDateTime())
  self.w("\\author")
  self.w("    Nicolas Delbosc")
  self.f:endComment()
end

-- generate the main() function
function MainGenerator:genMain()
  self.f:startFunction("int main()")

  --TODO:
  --self.f:comment("Choose the GPU for computations");
  --self.w("cudaSetDevice(2)");

  self.f:commentSection("Physical Properties")
  local params = model:genParametersList()
  for _, p in pairs(params) do
    self.f:comment(p.description)
    self.w(p:genDeclInit())
  end

  -- ALLOCATE GPU MEMORY FOR THE VELOCITY DFS ====================
  self.f:commentSection("Distribution Functions")
  self.f:commentAndLog("Allocate memory for the velocity distribution functions")
  self.w(memory:genMemAllocDfs())
  self.f:commentAndLog("Init distribution functions")
  -- loop on each axis
  for i, x in ipairs(model.position_name) do
    self.f:startForLoop(x, 0, model.size_names[i])
  end
  --TODO: restructure/rewrite these initialisations (it's a mess!)
  --The initialisation could be done in lua itself!
  -- generate each (basic) moment with their default values
  for _, moment in pairs(dynamics.moments) do
    for i, c in ipairs(moment:genDeclInit()) do
      if i <= node.D then
        self.w(c)
      end
    end
  end
  -- modify the moments using the initialisation method (if any)
  if self.initMethod then
    self.w(self.initMethod:genMoments())
  end
  if dynamics.genInitMoments then
    --compute the rest of the moments from rho and V (using MRT equibrium)
    for _, m in pairs(dynamics:genInitMoments()) do
      self.w(model.precision .. " " .. m)
    end
    -- compute the distribution functions using the invert transformation matrix
    local fis_mat = model.invM * model.moments_names
    for i = 1, node.Q do
      self.w(memory:genAccessToDf_CPU(i, model.position_name) .. " = " .. fis_mat[i][1])
    end
  else
    -- just set dfs to their equilibrium
    for i, fieq in ipairs(dynamics:genEquilibriumDfs()) do
      self.w(memory:genAccessToDf_CPU(i, model.position_name) .. " = " .. fieq)
    end
  end
  --close the for loops
  for i, x in ipairs(model.position_name) do
    self.f:endLoop()
  end
  self.f:commentAndLog("Upload the distribution functions to the GPU")
  self.w(memory:genMemCopyCpuToGpu())
  self.f:skipLine()
  self.f:commentAndLog("Allocate memory for the temporary distribution functions")
  self.w(memory:genMemAllocDfs_tmp())

  -- ALLOCATE GPU MEMORY TO STORE THE VELOCITY IF NEEDS TO BE =======
  if self.storeVelocity then
    self.f:skipLine()
    self.f:commentAndLog("Allocate memory for the velocity field")
    for _, v in pairs(model.velocity_name) do
      self.w(memory:genMemAllocGPU(v:upper() .. "_d"))
    end
  end
  -- ALLOCATE GPU MEMORY TO STORE THE DENSITY =======================
  if self.storeDensity then
    self.f:skipLine()
    self.f:commentAndLog("Allocate memory the store the density field")
    self.w(memory:genMemAllocGPU(model.density_name:upper() .. "_d"))
  end

  -- GENERATE THE CUDA THREAD LAYOUT ================================
  self.f:commentSection("Cuda Thread Layout")
  self.w(layout:genThreadBlockSize())
  self.w(layout:genThreadGridSize())

  -- LET EACH MODULE GENERATE ITS INITIALISATION CODE ===============
  for _, m in pairs(self.modules) do
    if m.genInit then
      self.w(m:genInit())
    end
  end

  -- START THE MAIN LOOP ============================================
  self.f:commentSection("Main Loop")
  self.w("bool running = true")
  self.w("unsigned int " .. model.time_name .. " = 0")
  exitConditions = {"running"} -- will store exit conditions given by the modules
  -- Get all the exit conditions for the modules
  for _, m in pairs(self.modules) do
    if m.genExitCondition then
      table.insert(exitConditions, m:genExitCondition())
    end
  end
  -- Start a while loop including all the exit conditions
  self.f:startWhileLoop(table.concat(exitConditions, " && "))

  -- CALL THE KERNEL ================================================
  args = {} -- additional arguments to the kernel call from the modules
  for _, m in pairs(self.modules) do
    if m.genKernelCallArgs then
      table.insert(args, m:genKernelCallArgs())
    end
  end
  if self.storeVelocity then -- add velocity pointers
    for _, v in pairs(model.velocity_name) do
      table.insert(args, memory:genPointer(v:upper() .. "_d"))
    end
  end
  if self.storeDensity then -- add density pointer
    table.insert(args, memory:genPointer(model.density_name:upper() .. "_d"))
  end
  args = table.concat(args, ", ")
  self.w(kernel:genCall(args))

  -- SWAP THE DISTRIBUTION FUNCTIONS ================================
  self.f:comment("Swap the distributions")
  self.w(memory:genSwap())
  self.w('getLastCudaError("' .. kernel.name .. '")')

  -- UPDATE THE MODULES =============================================
  for _, m in pairs(self.modules) do
    if m.genUpdate then
      if m.updateInterval and m.updateInterval > 1 then
        self.f:conditionalBlock(model.time_name .. "%" .. m.updateInterval .. "==0", m:genUpdate())
      elseif m.FPS then
        self.f:conditionalBlock(m.clock_name .. ".tick()", m:genUpdate())
      else
        self.w(m:genUpdate())
      end
    end
  end

  -- FINISH THE LOOP ================================================
  self.w(model.time_name .. "++")
  self.f:endLoop()

  -- FINISH THE PROGRAM =============================================
  self.f:log("End of program")
  self.f:endFunction()
end

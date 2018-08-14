-- Display the simulation using OpenGL
OpenGLRenderGenerator = class()
function OpenGLRenderGenerator:_init(args)
  args = args or {}
  --number of time-steps between each call to the update function
  -- (called at every time step if left empty)
  self.FPS = args.FPS or 30
  self.clock_name = "displayClock"
  -- which quantity to visualise
  self.quantity = args.quantity or "velocity_norm"
  -- check this is a know quantity
  if
    self.quantity ~= "velocity_norm" and self.quantity ~= "vorticity_norm" and
      self.quantity ~= "vorticity_norm_approximate"
   then
    error("Unknown quantity : " .. self.quantity)
  end
  -- for the vorticity, we need to store the velocity first
  -- (but not for the approximative version!)
  if self.quantity == "vorticity_norm" then
    main.storeVelocity = true
  end
  -- OPTIONAL PARAMETERS =================================
  self.min = args.min
  self.max = args.max
  if args.colors == "black_and_white" then
    self.colors = "ColorMode::BLACK_AND_WHITE"
  elseif args.colors == "rainbow" then
    self.colors = "ColorMode::RAINBOW"
  elseif args.colors == "diverging" then
    self.colors = "ColorMode::DIVERGING"
  end
  -- OPTIONAL PARAMETERS  FOR CPU RENDER =================
  -- render on CPU using OpenGLRenderCPU
  if args.mode == "CPU" then
    self.CPUmode = true
    self.renderclass = "OpenGLRenderCPU"
    -- ask the program to store the velocity
    main.storeVelocity = true
  else
    self.renderclass = "OpenGLRender"
  end
  self.velocity_resolution = args.velocity_resolution
  self.velocity_scale = args.velocity_scale

  if args.velocity_mode == "arrow" then
    self.velocity_mode = "VVM_ARROW"
  elseif args.velocity_mode == "line" then
    self.velocity_mode = "VVM_LINE"
  elseif args.velocity_mode == "arrow_norm" then
    self.velocity_mode = "VVM_ARROW_NORM"
  elseif args.velocity_mode == "line_norm" then
    self.velocity_mode = "VVM_LINE_NORM"
  elseif args.velocity_mode == "none" then
    self.velocity_mode = "NONE"
  end
end

function OpenGLRenderGenerator:genDependencies()
  return {
    self.renderclass .. ".hpp",
    "Time.h"
  }
end

function OpenGLRenderGenerator:genInit()
  inits = {
    self.renderclass .. " render(" .. table.concat(model.size_names, ", ") .. ")",
    "render.openWindow(800,800)"
  }
  if self.min then
    table.insert(inits, "render.setPlotMin(" .. self.min .. ")")
  end
  if self.max then
    table.insert(inits, "render.setPlotMax(" .. self.max .. ")")
  end
  if self.colors then
    table.insert(inits, "render.setColorMode(" .. self.colors .. ")")
  end
  if self.CPUmode then
    if self.velocity_resolution then
      table.insert(inits, "render.visu->setVelocityResolution(" .. self.velocity_resolution .. ")")
    end
    if self.velocity_scale then
      table.insert(inits, "render.visu->setVelocityScale(" .. self.velocity_scale .. ")")
    end
    if self.velocity_mode then
      if self.velocity_mode == "NONE" then
        table.insert(inits, "render.visu->disableVectorDisplay()")
      else
        table.insert(inits, "render.visu->setVelocityDisplayMode(" .. self.velocity_mode .. ")")
      end
    end
  end
  table.insert(inits, "TickingClock " .. self.clock_name .. "(" .. (1.0 / self.FPS) .. ")")
  return inits
end

function OpenGLRenderGenerator:genExitCondition()
  return "render.isOpen()"
end

function OpenGLRenderGenerator:genUpdate()
  if self.CPUmode then
    local velocities = {}
    for _, v in pairs(model.velocity_name) do
      table.insert(velocities, v:upper() .. "_d")
    end
    return {
      "render.downloadData(" .. table.concat(velocities, ", ") .. ")",
      "render.display()"
    }
  else
    return "render.display()"
  end
end

--generate arguments to add to the call of the compute kernel
function OpenGLRenderGenerator:genKernelCallArgs()
  return "render.gpu_ptr()"
end

--generate arguments to add to the definition of the compute kernel
function OpenGLRenderGenerator:genKernelDefArgs()
  return model.precision .. " *plot"
end

--generate the code to be added to the compute kernel
function OpenGLRenderGenerator:genKernelCode()
  local quantity = ""
  if self.quantity == "velocity_norm" then
    -- compute the norm of the velocity
    local vx2 = model:genVelocity() .. model:genVelocity()
    quantity = "sqrtf(" .. vx2 .. ")"
  elseif self.quantity == "density" then
    quantity = model.density_name
  elseif self.quantity == "vorticity_norm" or self.quantity == "vorticity_norm_approximate" then
    -- compute the norm of the vorticity
    -- TODO: only works for periodic and only 2D
    if (node.D == 3) then
      error("Vorticity Computation is only available in 2D for now.")
    end
    local X = model.position_name
    local x = X[1]
    local y = X[2]
    local xplus = x .. sign(1)
    local xminus = x .. sign(-1)
    local yplus = y .. sign(1)
    local yminus = y .. sign(-1)
    if self.quantity == "vorticity_norm" then
      local VX = model.velocity_name[1]:upper()
      local VY = model.velocity_name[2]:upper()
      local DvyDx = memory:genAccess_GPU(VY, xplus, y) - memory:genAccess_GPU(VY, xminus, y)
      local DvxDy = memory:genAccess_GPU(VX, x, yplus) - memory:genAccess_GPU(VX, x, yminus)
      quantity = (DvyDx - DvxDy) / 2
    else -- approximate version
      local dirs = {
        vector({1, 0}),
        vector({-1, 0}),
        vector({0, 1}),
        vector({0, -1})
      }
      quantity = "-4.f/9.f"
      for _, dir in pairs(dirs) do
        for i, ei in ipairs(node.directions) do
          if dir ^ ei == 1 and dir .. ei == 0 then
            quantity = quantity + memory:genAccess(i, x .. sign(dir[1]), y .. sign(dir[2]))
          end
        end
      end
    end
  end
  return memory:genAccess_GPU("plot", model.position_name) .. " = " .. quantity
end

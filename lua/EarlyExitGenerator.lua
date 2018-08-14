-- exit the simulation if it crashes
EarlyExitGenerator = class()

function EarlyExitGenerator:_init(args)
end

function EarlyExitGenerator:genDependencies()
  return "KernelStatus.hpp"
end

function EarlyExitGenerator:genInit()
  return "KernelStatus kernelStatus"
end

function EarlyExitGenerator:genExitCondition()
  return "(kernelStatus.getStatus()==KernelStatus::OK)"
end

function EarlyExitGenerator:genKernelCallArgs()
  return "kernelStatus.gpu_ptr()"
end

function EarlyExitGenerator:genKernelDefArgs()
  return "KernelStatus::Enum *status"
end

function EarlyExitGenerator:genKernelCode()
  return "if ( " .. model.density_name .. " <= 0 ) \
    *status = KernelStatus::UNSTABLE"
end

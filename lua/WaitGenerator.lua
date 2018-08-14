-- Very simple module to pause the program for some duration between each time step
-- (useful for debugging)
WaitGenerator = class()
function WaitGenerator:_init(args)
  args = args or {}
  --number of time-steps between each call to the update function
  self.updateInterval = args.interval or 0
  -- duration of the wait
  self.duration = args.duration or 0.1
end

function WaitGenerator:genDependencies()
  return "Time.h"
end

function WaitGenerator:genUpdate()
  return "wait_time(" .. self.duration .. ")"
end

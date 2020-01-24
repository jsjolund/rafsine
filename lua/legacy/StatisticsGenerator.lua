-- TODO: define a general 'module' class with
--       NO NEED for a module, but every module should/can have
--       genDependencies()
--       genInit()
--       genUpdate()
--       genExitCondition()

-- Display statistics information (using Statistics.h)
StatisticsGenerator = class()
function StatisticsGenerator:_init(args)
  self.delay = args.delay or 1.0
  self.outputMode = args.outputMode or "same_line"
end

function StatisticsGenerator:genDependencies()
  return "Statistics.h"
end

function StatisticsGenerator:genInit()
  local init = {"Statistics stats(" .. model:genNumberOfCells() .. ", " .. self.delay .. ")"}
  if self.outputMode == "new_line" then
    table.insert(init, "stats.setOutputMode(PrintMode::NEW_LINE);")
  end
  return init
end

function StatisticsGenerator:genUpdate()
  return "stats.update()"
end

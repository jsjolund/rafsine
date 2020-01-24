-- End of line
-- (defined only in this file)
local endl = "\n"

FileWriter = class()
function FileWriter:_init(file_name)
  -- Name of the file
  self.file_name = file_name
  -- open the file for wrtting and store an handle
  self.file = io.open(self.file_name, "w")
  if not self.file then
    error("Error while opening " .. self.file_name .. " for writting.\nCheck that the folder does exist.")
  end
  -- text used for indentation
  self.space = "  "
  -- current indentation
  self.indent = ""
  --current mode of writting ('code', 'comment', 'macro' )
  self.mode = "code"
  -- short for self.file:write(self.indent.."...")
  self.w = function(line)
    self.file:write(self.indent .. line)
  end
end

-- Generate the inclusion of the dependencies
-- ex: std::cout
function FileWriter:genDependencies()
  self.w("#include <iostream>" .. endl)
  self:writeLine("using std::cout")
  self:writeLine("using std::endl")
end

--increase the indentation
function FileWriter:increaseIndentation()
  self.indent = self.indent .. self.space
end

--reduce the indentation
function FileWriter:decreaseIndentation()
  self.indent = string.sub(self.indent, 1, -(string.len(self.space) + 1))
end

-- Write a line (indent and add ';' if necessary)
function FileWriter:writeLine(line)
  if type(line) ~= "string" then
    error("Wrong type of input")
  end
  if self.mode == "code" then
    self.w(line .. ";" .. endl)
  elseif self.mode == "comment" then
    self.w("* " .. line .. endl)
  elseif self.mode == "macro" then
    self.file:write(line .. endl)
  end
end

-- Write some text (either a single line or multiple lines in a table)
function FileWriter:write(text)
  if type(text) == "string" then
    self:writeLine(text)
  elseif type(text) == "table" then
    for _, line in pairs(text) do
      self:writeLine(line)
    end
  end
end

-- Write a block of text inside { } and with increased indentation
function FileWriter:writeBlock(text)
  self.w("{" .. endl)
  self:increaseIndentation()
  self:write(text)
  self:decreaseIndentation()
  self.w("}" .. endl)
end

-- Add an empty line
function FileWriter:skipLine()
  self.file:write(endl)
end

-- generate an include line
function FileWriter:include(fileName)
  if type(fileName) == "string" then
    self.w('#include "' .. fileName .. '"' .. endl)
  elseif type(fileName) == "table" then
    for _, name in pairs(fileName) do
      self:include(name)
    end
  end
end

-- Start a function
function FileWriter:startFunction(declaration)
  self.w(declaration .. endl)
  self.w("{" .. endl)
  self:increaseIndentation()
end

-- Start a kernel function
function FileWriter:startKernel(declaration)
  self.w("__global__ " .. declaration .. endl)
  self.w("{" .. endl)
  self:increaseIndentation()
end

-- End a function
function FileWriter:endFunction()
  self:decreaseIndentation()
  self.w("}" .. endl)
end

-- generate a basic one line comment
function FileWriter:comment(comment)
  if (comment ~= "") then
    self.w("//" .. comment .. endl)
  end
end
-- generate a comment line (designed to separate sections of code)
function FileWriter:commentSection(sectionName)
  local section_size = 50 -- number of characters of the line
  local filling = "="
  self:skipLine()
  self.w(
    "//=== " .. sectionName:upper() .. " " .. filling:rep(section_size - sectionName:len() - self.indent:len()) .. endl
  )
end

-- Start a multiline comment
function FileWriter:startComment(description)
  description = description or ""
  self.w("/** " .. description .. endl)
  self:increaseIndentation()
  self.mode = "comment"
end

-- End a multiline comment
function FileWriter:endComment()
  self.w("*/" .. endl)
  self:decreaseIndentation()
  self.mode = "code"
end

-- Start the macro mode
function FileWriter:startMacros(description)
  if description then
    self:comment(description)
  end
  self.mode = "macro"
end

-- End the macro mode
function FileWriter:endMacros()
  self.mode = "code"
end

-- Start a for loop
-- suppose index is int, increment is ++
--TODO: define a function prototype that would do {, increaseIndentation, and closing automatically
function FileWriter:startForLoop(index, min, max)
  self.w("for(int " .. index .. "=" .. min .. "; " .. index .. "<" .. max .. "; " .. index .. "++)" .. endl)
  self.w("{" .. endl)
  self:increaseIndentation()
end

-- End a loop
function FileWriter:endLoop()
  self:decreaseIndentation()
  self.w("}" .. endl)
end

-- Start a while loop
function FileWriter:startWhileLoop(condition)
  self.w("while (" .. condition .. ")" .. endl)
  self.w("{" .. endl)
  self:increaseIndentation()
end

-- if {} else {} block generation with single line or multiline mode
function FileWriter:conditionalBlock(condition, statementIfTrue, statementIfFalse)
  self.w("if (" .. condition .. ")" .. endl)
  -- write statement if condition is true
  if type(statementIfTrue) == "string" then --one line statement
    self:writeLine(self.space .. statementIfTrue)
  elseif type(statementIfTrue) == "table" then --multiple lines statement
    self:writeBlock(statementIfTrue)
  else
    error("unknown statement type given (" .. type(statementIfTrue) .. ") when condition (" .. condition .. ") is true")
  end
  -- write statement if condition is false (optional)
  if (statementIfFalse) then
    self.w("else" .. endl)
    if type(statementIfFalse) == "string" then --one line statement
      self.writeLine(self.space .. statementIfFalse)
    elseif type(statementIfFalse) == "table" then --multiple lines statement
      self:writeBlock(statementIfFalse)
    end
  end
end

-- switch generations
function FileWriter:switchBlock(params)
  self.w("switch(" .. params.expression .. ")" .. endl)
  self.w("{" .. endl)
  self:increaseIndentation()
  for i, c in ipairs(params.cases) do
    if params.comments then
      self:comment(params.comments[i])
    end
    self.w("case " .. c .. ":" .. endl)
    self:increaseIndentation()
    self:write(params.statements[i])
    self:write("break")
    self:decreaseIndentation()
  end
  if params.default then
    self.w("default:" .. endl)
    self:increaseIndentation()
    self:write(params.default)
    self:decreaseIndentation()
  end
  self:decreaseIndentation()
  self.w("}" .. endl)
end

-- Text to be displayed in the generated language console
function FileWriter:log(text)
  self:writeLine('cout << "' .. text .. '" << endl')
end

-- Text to be added as a comment and logged in the destination language
function FileWriter:commentAndLog(text)
  self:comment(text)
  self:log(text)
end

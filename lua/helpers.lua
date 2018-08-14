-- Set of simple functions, but often reused ones
local lfs = require "lfs"

--returns the current date and time
function currentDateTime()
  return os.date("%A %d %B %Y, %H:%M")
end

--check that a folder exists, and create one if it does not
function checkFolder(folder_name)
  local attr = lfs.attributes(folder_name)
  if attr == nil then -- folder does not exist
    --print("creating directory", folder_name)
    lfs.mkdir(folder_name)
  else
    if attr.mode == "file" then
      error("folder's name '" .. folder_name .. "' is already used by a file.")
    end
  end
end

-- Check the type of a variable, return a error if not correct
function assertType(variable, type_)
  assert(type(variable) == type_, "Expected a " .. type_ .. " got a " .. type(variable) .. ".")
end
function assertTable(variable)
  assertType(variable, "table")
end

-- add a string at the end of each element of a table
function append(a_table, a_string)
  for _, elt in pairs(a_table) do
    elt = elt .. a_string
  end
end

-- Expand a table for listing,printing...
function expand(a_table)
  if type(a_table) == "number" then
    return a_table
  end
  return table.concat(a_table, ", ")
end

function dump(o)
  if type(o) == 'table' then
     local s = '{ '
     for k,v in pairs(o) do
        if type(k) ~= 'number' then k = '"'..k..'"' end
        s = s .. '['..k..'] = ' .. dump(v) .. ','
     end
     return s .. '} '
  else
     return tostring(o)
  end
end

-- convert a floating point number to an integer with proper rounding rules
function round(number)
  int = math.floor(number)
  if number - int > 0.5 then
    int = int + 1
  end
  return int
end

-- print a matrix to the console
function printMatrix(M)
  for _, mi in pairs(M) do
    for _, mij in pairs(mi) do
      sp = ""
      if type(mij) == "number" then
        if mij == 0 then
          mij = 0 -- transforms -0 into 0
        end
        if mij >= 0 then
          sp = " "
        end
        -- use only 4 decimals after 0.
        mij = math.floor(mij * 10000) / 10000
      elseif type(mij) == "string" then
        if string.sub(mij, 1, 1) ~= "-" then
          sp = " "
        end
      end
      local spaces = string.rep(" ", 10 - tostring(mij):len() - sp:len())
      io.write(sp .. mij .. spaces)
    end
    io.write("\n")
  end
end

-- compute the norm2 of a table (can be a vector)
function norm(a_table)
  local norm = 0
  for i, val in ipairs(a_table) do
    norm = norm + val * val
  end
  return math.sqrt(norm)
end

-- fill a table
function fill(a_table, value)
  for i = 1, table.getn(a_table) do
    a_table[i] = value
  end
end

-- generate a table
function genTable(size, value)
  local a_table = {}
  for i = 1, size do
    a_table[i] = value
  end
  return a_table
end

-- extract a column from a 2D array and set the right operators on it
function COL(array, col_num)
  local pl = require "pl.array2d"
  local col = array2d.column(array, col_num)
  setmetatable(col, metaTableColumns)
  return col
end

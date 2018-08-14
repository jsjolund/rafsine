-- merge 2 tables
mergeTables = function(table1, table2)
  --if type(table2)=='string' then return append(table1, table2) end
  result = {}
  for i, e1 in ipairs(table1) do
    local e2 = table2[i]
    if (e2 ~= nil) then
      table.insert(result, e1 .. e2)
    end
  end
  return result
end

-- Add a string after each element of a table
append = function(table1, string)
  result = {}
  for _, e1 in ipairs(table1) do
    table.insert(result, e1 .. string)
  end
  return result
end

-- Add a string before each element of a table
prepend = function(string, table2)
  result = {}
  for _, e2 in ipairs(table2) do
    table.insert(result, string .. e2)
  end
  return result
end

-- Return true if the object is a scalar (string or number)
isScalar = function(x)
  return (type(x) == "string" or type(x) == "number")
end

-- Function to merge 2 elements,
-- call mergeTables, append or prepend accordingly
mergeTwo = function(elt1, elt2)
  if isScalar(elt1) and isScalar(elt2) then
    return elt1 .. elt2
  elseif isScalar(elt1) and type(elt2) == "table" then
    return prepend(elt1, elt2)
  elseif type(elt1) == "table" and isScalar(elt2) then
    return append(elt1, elt2)
  elseif type(elt1) == "table" and type(elt2) == "table" then
    return mergeTables(elt1, elt2)
  end
end

-- function to merge two (or more) tables (of text)
-- concatenate line by line
-- can also add a string to each element of a table

merge = function(tables)
  t = table.remove(tables, 1)
  if table.getn(tables) == 0 then
    return t
  end
  --if table.getn(tables)==1 and type(tables[1])=='string' then return append(t,tables[1]) end
  --if type(t)=='string' then return prepend(t, merge(tables)) end
  --return mergeTables( t, merge(tables))
  return mergeTwo(t, merge(tables))
end

-- Function to transform a non-table object to a one-line table object
-- basically add { } around the object
--           do nothing if already a table
function totable(X)
  if type(X) == "table" then
    return X
  else
    return {X}
  end
end

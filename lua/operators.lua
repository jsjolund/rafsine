--TODO: use isScalar() function defined in merge.lua

-- function to check if a number is an integer
function isInt(x)
  return (x - math.floor(x)) == 0
end

--function to compute the scalar product of Ei and U=(Vx,Vy,Vz)
scalar_product = function(Ei, U)
  --string to display
  local result = ""
  --number of non-zero components to decide if parenthesis are needed
  local components = 0
  for j, Eij in ipairs(Ei) do
    if ((type(Eij) == "number") and (Eij ~= 0)) then
      components = components + 1
      if (Eij == 1) then --add this component
        -- if it's the first component, no need for '+'
        if (components == 1) then
          result = result .. " " .. U[j]
        else
          result = result .. "+" .. U[j]
        end
      elseif (Eij == -1) then -- substract this component
        result = result .. "-" .. U[j]
      else -- general component ============================
        -- if it's the first component, no need for '+'
        if (components == 1) then
          result = result .. " " .. tostring(Eij) .. "*" .. U[j]
        else
          result = result .. "+" .. tostring(Eij) .. "*" .. U[j]
        end
      end
    elseif (type(Eij) == "string") then --both components are string
      components = components + 1
      if (components == 1) then
        result = result .. Eij .. "*" .. U[j]
      else
        result = result .. "+" .. Eij .. "*" .. U[j]
      end
    end
  end
  -- Add brackets if not empty
  if (result == "") then
    return result
  else
    return "(" .. result .. ")"
  end
end

-- function compute the sum of a and b
-- where a and b can be either a number or a string
sum = function(a, b)
  -- if they are both numbers
  if type(a) == "number" and type(b) == "number" then
    return a + b
  end
  -- if both are strings
  if type(a) == "string" and type(b) == "string" then
    -- if one is empty, then return the other one
    if a == "" then
      return b
    end
    if b == "" then
      return b
    end
    -- if b starts with minus, don't use the + sign
    if b:sub(1, 1) == "-" then
      return a .. b
    end
    return a .. "+" .. b
  end
  -- if first is number
  if type(a) == "number" and type(b) == "string" then
    if a == 0 then
      return b
    end
    if b == "" then
      return a
    end
    -- if b starts with minus, don't use the + sign
    if b:sub(1, 1) == "-" then
      return tostring(a) .. b
    end
    return tostring(a) .. "+" .. b
  end
  -- if second is number
  if type(a) == "string" and type(b) == "number" then
    return sum(b, a)
  end
end

-- function compute the subtraction  a - b
-- where a and b can be either a number or a string
sub = function(a, b)
  if type(b) == "number" then
    return sum(a, -b)
  end
  if type(b) == "string" then
    --if the string is and addition or substraction, then add brackets
    if string.find(b, "+") ~= nil or string.find(b, "-") ~= nil then
      b = "(" .. b .. ")"
    end
    if b:sub(1, 1) == "-" then --remove the minus sign
      return sum(a, string.gsub(b, 2, -1))
    else -- add a minus sign
      return sum(a, "-" .. b)
    end
  end
end

-- function compute the product of a and b
-- where a and b can be either a number or a string
-- if a or b is a table, call vector_multiply
prod = function(a, b)
  -- if a or b is a vector
  if type(a) == "table" or type(b) == "table" then
    return vector_multiply(a, b)
  end
  -- if they are both numbers
  if type(a) == "number" and type(b) == "number" then
    return a * b
  end
  -- if both are strings
  if type(a) == "string" and type(b) == "string" then
    --if the string is and addition, then add brackets
    if string.find(a, "+") ~= nil or string.find(a, "-") ~= nil then
      a = "(" .. a .. ")"
    end
    if string.find(b, "+") ~= nil or string.find(b, "-") ~= nil then
      b = "(" .. b .. ")"
    end
    return a .. "*" .. b
  end
  -- if first is number
  if type(a) == "number" and type(b) == "string" then
    if string.find(b, "+") ~= nil or string.find(b, "-") ~= nil then
      b = "(" .. b .. ")"
    end
    if a == 0 then
      return 0
    end
    if a == 1 then
      return b
    end
    if a == -1 then
      return "-" .. b
    end
    return tostring(a) .. "*" .. b
  end
  -- if second is number
  if type(a) == "string" and type(b) == "number" then
    return prod(b, a)
  end
end

-- function to divide a per b
divide =
  function(a, b)
  -- if a or b is a table, use vector_divide
  if type(a) == "table" or type(b) == "table" then
    return vector_divide(a, b)
  end
  -- if they are both numbers
  if type(a) == "number" and type(b) == "number" then
    return a / b
  end
  -- if both are strings
  if type(a) == "string" and type(b) == "string" then
    --if the string is and addition, then add brackets
    if string.find(a, "+") ~= nil or string.find(a, "-") ~= nil then
      a = "(" .. a .. ")"
    end
    if string.find(b, "+") ~= nil or string.find(b, "-") ~= nil then
      b = "(" .. b .. ")"
    end
    return a .. "/" .. b
  end
  -- if first is number
  if type(a) == "number" and type(b) == "string" then
    if
      string.find(b, "+") ~= nil or string.find(b, "-") ~= nil or string.find(b, "*") ~= nil or
        string.find(b, "/") ~= nil
     then
      b = "(" .. b .. ")"
    end
    if a == 0 then
      return 0
    end
    return tostring(a) .. "/" .. b
  end
  -- if second is number
  if type(a) == "string" and type(b) == "number" then
    if string.find(a, "+") ~= nil or string.find(a, "-") ~= nil then
      a = "(" .. a .. ")"
    end
    if b == 0 then
      return "nan"
    end
    if b == 1 then
      return a
    end
    if b == -1 then
      return "-" .. a
    end
    return a .. "/" .. tostring(b)
  end
end

-- function to define the exponent
-- only works for b integer >= 0
pow = function(a, b)
  if type(b) == "number" then
    if (b == 0) then
      if type(a) == "table" then
        local result = genTable(table.getn(a), 1)
        setmetatable(result, getmetatable(a))
        return result
      else
        return 1
      end
    end
    if (b == 1) then
      return a
    end
    local result = a
    for i = 2, b do
      result = prod(result, a)
    end
    if type(a) == "table" then
      setmetatable(result, getmetatable(a))
    end
    return result
  else
    error("Error ", b, " should be a positive integer")
  end
end

--function to compute the scalar product of U=(Ux,Uy,Uz) and V=(Vx,Vy,Vz)
scalar_product = function(U, V)
  if V == nil then
    error("Right hand side of scalar operator is nil")
  end
  return SUM(vector_multiply(U, V))
end

--function to multiply a vector by a scalar
scalar_multiply = function(alpha, U)
  --print("using scalar_multiply")
  --store the multiplication of each component
  local res = vector()
  -- compute all the multiplications
  for i, Ui in ipairs(U) do
    res[i] = prod(alpha, Ui)
  end
  return res
end

-- function to multiply each component of two vectors
vector_multiply = function(U, V)
  if type(U) == "number" or type(U) == "string" then
    return scalar_multiply(U, V)
  end
  if type(V) == "number" or type(V) == "string" then
    return scalar_multiply(V, U)
  end
  --print("using vector_multiply", "type of U:", type(U), "type of V:", type(V))
  --store the multiplication of each component
  local mults = vector()
  -- compute all the multiplications
  for i, Ui in ipairs(U) do
    if V[i] ~= nil then
      mults[i] = prod(Ui, V[i])
    end
  end
  return mults
end

-- function to compute the vector product of 2 vectors
vector_product = function(U, V)
  local w3 = U[1] * V[2] - U[2] * V[1]
  if U[3] then --3D vector
    local w1 = U[2] * V[3] - U[3] * V[2]
    local w2 = U[3] * V[1] - U[1] * V[3]
    return vector({w1, w2, w3})
  else --2D vector
    return w3
  end
end

--function to divide a vector by a scalar
scalar_divide = function(U, alpha)
  if (type(U) ~= "table") then
    print("Error: scalar_divide(U,alpha), U must be a vector (table)")
    return "nan"
  end
  if (type(alpha) ~= "number" and type(alpha) ~= "string") then
    print("Error: scalar_divide(U,alpha), alpha must be a number or a string")
    return "nan"
  end
  --store the division of each component
  local res = vector()
  -- compute all the division
  for i, Ui in ipairs(U) do
    res[i] = divide(Ui, alpha)
  end
  return res
end

--function to divide each component of two vectors (of the same size)
vector_divide = function(U, V)
  if type(U) == "number" or type(U) == "string" then
    error("Error: vector_divide(U,V), U must be a vector (table)")
    return "nan"
  end
  if type(V) == "number" or type(V) == "string" then
    return scalar_divide(U, V)
  end
  print("using vector_divide", "type of U:", type(U), "type of V:", type(V))
  --store the division of each component
  local divs = vector()
  -- compute all the divisions
  for i, Ui in ipairs(U) do
    if V[i] ~= nil then
      divs[i] = divide(Ui, V[i])
    end
  end
  return divs
end

--function to add two vectors U=(Ux,Uy,Uz) and V=(Vx,Vy,Vz)
add_vector = function(U, V)
  --store the addition of each component
  local sums = vector()
  -- compute all the additions
  for i, Ui in ipairs(U) do
    if V[i] ~= nil then
      sums[i] = sum(Ui, V[i])
    end
  end
  setmetatable(sums, getmetatable(U))
  return sums
end

--function to compute the difference of U=(Ux,Uy,Uz) and V=(Vx,Vy,Vz)
sub_vector = function(U, V)
  --store the diffence of each component
  local diff = vector()
  -- compute all the differences
  for i, Ui in ipairs(U) do
    if V[i] ~= nil then
      diff[i] = sub(Ui, V[i])
    end
  end
  setmetatable(diff, getmetatable(U))
  return diff
end

--function to compute the opposite of a vector
negate_vector = function(U)
  local negU = vector()
  for i, Ui in ipairs(U) do
    negU[i] = sub(0, Ui)
  end
  return negU
end

--function to compare two vectors
compare_vector = function(U, V)
  if #U ~= #V then
    return false
  end
  for i, Ui in ipairs(U) do
    if Ui ~= V[i] then
      return false
    end
  end
  return true
end

debug.getmetatable("").__add = sum
debug.getmetatable("").__sub = sub
debug.getmetatable("").__mul = prod
debug.getmetatable("").__div = divide
debug.getmetatable("").__pow = pow

--create a metatable for vectors operators
metaTableVectors = {
  --scalar product of 2 vectors with the operator ..
  __concat = scalar_product,
  -- multiply two vectors (or one vector and one number)
  __mul = vector_multiply,
  -- mathematical vector product of 2 vectors
  __pow = vector_product,
  -- divide two vectors (or one vector by one number)
  __div = vector_divide,
  -- sum of two vectors
  __add = add_vector,
  -- difference of two vectors
  __sub = sub_vector,
  -- negate a vector
  __unm = negate_vector,
  -- compare two vectors components
  __eq = compare_vector
}

--create a metatable for operating on whole column
metaTableColumns = {
  -- multiply two column (each component)
  __mul = vector_multiply,
  -- compute the power of each component
  -- NOTE: this is the main difference with metaTableVectors
  --       this would be the cross product of 2 vectors
  __pow = pow,
  -- divide two columns (or one column by one number)
  __div = vector_divide,
  -- sum of two columns
  __add = add_vector,
  -- difference of two columns
  __sub = sub_vector,
  -- negate a columns
  __unm = negate_vector,
  -- compare two columns (each components)
  __eq = compare_vector
}

--create a vector and set its operators as metaTableVectors
--so a vector is a lua table with vectors operators
vector = function(table_values)
  local res = {}
  if type(table_values) == "table" then
    for key, value in pairs(table_values) do
      res[key] = value
    end
  end
  setmetatable(res, metaTableVectors)
  return res
end

-- Returns a table where each element is replace by its absolute value
-- TODO: make it work for arithmetic expressions too
ABS = function(a_table)
  local res = {}
  for key, value in pairs(a_table) do
    res[key] = math.abs(value)
  end
  return res
end

-- SUM function that sum all the elements of a table, (or only for some indices TODO)
SUM = function(a_table)
  if type(a_table) ~= "table" then
    return a_table
  end
  local sum
  local counter = 0
  for _, elt in pairs(a_table) do
    if counter == 0 then
      sum = elt
    else
      sum = sum + elt
    end
    counter = counter + 1
  end
  return sum
end
-- Compute the product of all the elements of a table
PROD = function(a_table)
  if type(a_table) ~= "table" then
    return a_table
  end
  -- compute the product of all the elements in the table
  local prod
  local counter = 0
  for _, elt in pairs(a_table) do
    if counter == 0 then
      prod = elt
    else
      prod = prod * elt
    end
    counter = counter + 1
  end
  return prod
end

--function to return the sign of a number as a text
--works with vectors
function sign(X)
  if type(X) == "number" then
    if X == 0 then
      return "     "
    elseif X > 0 then
      return "plus "
    else
      return "minus"
    end
  elseif type(X) == "table" then
    local signs = {}
    for i, Xi in ipairs(X) do
      signs[i] = sign(Xi)
    end
    return signs
  end
end

--function to return the sign of a number as a text
--works with vectors
function floor(X)
  if type(X) == "number" then
    return math.floor(X)
  elseif type(X) == "table" then
    local vec = {}
    for i, Xi in ipairs(X) do
      vec[i] = math.floor(Xi)
    end
    return vec
  end
end

-- generate a ternary operator like
-- (x==LatSizeX-1) ? (0) : (x+1);
function ternary(condition, onTrue, onFalse)
  return "((" .. condition .. ") ? (" .. onTrue .. ") : (" .. onFalse .. "))"
end

-- compute the norm of a vector
-- LNORM states what Ln norm to use
-- LNORM = 0 : norm returns 1
-- LNORM = 1 : norm returns SUM(ABS(vi))
-- LNORM = 2 : norm returns SQRT(SUM(vi*vi))
function norm(v, Ln)
  Ln = Ln or 2
  if Ln == 0 then
    return 1
  elseif Ln == 1 then
    return SUM(ABS(v))
  elseif Ln == 2 then
    return math.sqrt(v .. v)
  else
    error("Unknown Ln value.", Ln)
  end
end

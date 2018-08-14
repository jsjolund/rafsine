-- Define the D3Q27 node

-- directions sorted by
-- zero, 6 main axis, 12 sub-diagonals, 8 diagonals
local dirs = {{}, {}, {}, {}}
-- weights associated with each axis
local w = {"8/27", "2/27", "1/54", "1/216"}
for i = -1, 1 do
  for j = -1, 1 do
    for k = -1, 1 do
      local ei = vector({i, j, k})
      local wi
      --[[
      if ei..ei==0 then wi = "8/27" end
      if ei..ei==1 then wi = "2/27" end
      if ei..ei==2 then wi = "1/57" end
      if ei..ei==3 then wi = "1/256" end
      table.insert(directions, ei)
      table.insert(weights, wi)
      --]]
      table.insert(dirs[1 + (ei .. ei)], ei)
    end
  end
end

-- sort the directions
local directions = {}
local weights = {}
for norm2 = 1, 4 do
  for _, ei in pairs(dirs[norm2]) do
    table.insert(directions, ei)
    table.insert(weights, w[norm2])
  end
end

--[[
for i = 1,27 do
  print("("..expand(directions[i])..")", weights[i] )
end
error("end")
--]]
D3Q27Descriptor = NodeDescriptor(3, 27, directions, weights)

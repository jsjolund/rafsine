-- Generate various kind of documentations for the code

-- Generate a diagram of the node velocities using latex and tikz-3dplot package
function generateNodeDiagram(node)
  -- parameters for drawing the velocity vectors
  local pline = "very thick,->"
  local scale = node.D
  -- folder where the diagram is generated
  local folder = doc_destination_folder or "."
  folder = folder .. "/nodes"
  checkFolder(folder)
  local DdQq = "D" .. node.D .. "Q" .. node.Q
  local file_name = folder .. "/" .. DdQq .. ".tex"
  print("Generating node diagram '" .. file_name .. "' ...")
  local file = io.open(file_name, "w")
  --write some text and open a new line
  local function w(text)
    file:write(text .. "\n")
  end
  file:write(
    "%Generated node velocities diagram\n%Nicolas Delbosc\n%University of Leeds\n" ..
      "\\documentclass{article}" ..
        "\n" ..
          "\\usepackage{verbatim}" ..
            "\n" ..
              "\\usepackage{tikz}" ..
                "\n" ..
                  "\\usepackage{3dplot}" ..
                    "\n" ..
                      "\\usepackage[active,tightpage]{preview}" ..
                        "\n" ..
                          "\\PreviewEnvironment{tikzpicture}" ..
                            "\n" .. "\\setlength\\PreviewBorder{2mm}" .. "\n" .. "\\begin{document}" .. "\n"
  )

  --set the plot display orientation
  --syntax: \tdplotsetdisplay{\theta_d}{\phi_d}
  if node.D == 2 then
    w("\\tdplotsetmaincoords{0}{0}")
  else
    w("\\tdplotsetmaincoords{80}{30}")
  end

  --start tikz picture, and use the tdplot_main_coords style to implement the display
  --coordinate transformation provided by 3dplot
  w("\\begin{tikzpicture}[scale=" .. scale .. ",tdplot_main_coords]")

  --set up some coordinates
  --file:write("\\coordinate (O) at (0,0,0);\n")
  w("\\coordinate (O) at (0,0,0);")

  --draw figure contents--
  ------------------------

  -- draw axis
  w("\\draw[very thick,->] (2,0,0) -- (3,0,0) node[anchor=north east]{$x$};")
  w("\\draw[very thick,->] (2,0,0) -- (2,1,0) node[anchor=north east]{$y$};")
  if node.D == 3 then
    w("\\draw[very thick,->] (2,0,0) -- (2,0,1) node[anchor=north east]{$z$};")
  end

  --draw an outline box
  w("\\draw[dashed] (-1,-1,0) -- ( 1,-1,0) -- (1,1,0) -- (-1,1,0) -- cycle;")
  if node.D == 3 then
    w("\\draw[dashed] (-1,-1,0) -- ( 1,-1,0) -- (1,1,0) -- (-1,1,0) -- cycle;")
    w("\\draw[dashed] (0,-1,-1) -- ( 0,-1,1) -- (0,1,1) -- (0,1,-1) -- cycle;")
    w("\\draw[dashed] (-1,0,-1) -- ( -1,0,1) -- (1,0,1) -- (1,0,-1) -- cycle;")
    w("\\draw[thick] (-1,-1,-1) -- ( 1,-1,-1) -- (1,1,-1) -- (-1,1,-1) -- cycle;")
    w("\\draw[thick] (-1,-1,1) -- ( 1,-1,1) -- (1,1,1) -- (-1,1,1) -- cycle;")
    w("\\draw[thick] (-1,-1,1) -- (-1,-1,-1);")
    w("\\draw[thick] (1,-1,1) -- (1,-1,-1);")
    w("\\draw[thick] (1,1,1) -- (1,1,-1);")
    w("\\draw[thick] (-1,1,1) -- (-1,1,-1);")
  end

  if node.D == 2 then
    --draw each direction
    for i = 1, node.Q do
      local ei = node.directions[i]
      local P = "(" .. ei[1] .. "," .. ei[2] .. ",0)"
      if (ei .. ei == 0) then
        -- display as a dot
        w("\\node at (0,0) [circle,fill=black] {};")
        w("\\draw[] (0,0,0) node[anchor=north east]{$\\vec{e_{" .. (i - 1) .. "}}$};")
      else
        -- relative position
        local rel_pos
        if ei[2] > 0 then
          rel_pos = "south "
        else
          rel_pos = "north "
        end
        if ei[1] > 0 then
          rel_pos = rel_pos .. "west"
        else
          rel_pos = rel_pos .. "east"
        end
        -- display as vectors
        w("\\draw[" .. pline .. "] (O) -- " .. P .. " node[anchor=" .. rel_pos .. "]{$\\vec{e_{" .. (i - 1) .. "}}$};")
      end
    end
  else -- 3D node
    for i = 1, node.Q do
      local ei = node.directions[i]
      local P = "(" .. ei[1] .. "," .. ei[2] .. "," .. ei[3] .. ")"
      if (ei .. ei == 0) then
        -- display as a dot
        w("\\node at (0,0) [circle,fill=black] {};")
        w("\\draw[] (0,0,0) node[anchor=north east]{$\\vec{e_{" .. (i - 1) .. "}}$};")
      else
        -- relative position
        local rel_pos
        if ei[3] > 0 then
          rel_pos = "south "
        else
          rel_pos = "north "
        end
        if (ei[1] > 0) or (ei[1] + ei[2] > 0) then
          rel_pos = rel_pos .. "west"
        else
          rel_pos = rel_pos .. "east"
        end
        -- display as vectors
        w("\\draw[" .. pline .. "] (O) -- " .. P .. " node[anchor=" .. rel_pos .. "]{$\\vec{e_{" .. (i - 1) .. "}}$};")
      end
    end
  end

  w("\\end{tikzpicture}")
  w("\\end{document}")
  io.close(file)
  os.execute("pdflatex -interaction=batchmode " .. file_name)
  os.execute("rm " .. DdQq .. ".aux")
  os.execute("rm " .. DdQq .. ".log")
  os.execute("mv " .. DdQq .. ".pdf " .. folder)
  print("Node diagram finished")
end

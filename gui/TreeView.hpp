#pragma once

#include <FL/Fl.H>
#include <FL/Fl_Tree.H>
#include <iostream>

#include "../ext/st_tree/include/st_tree.h"
#include "../sim/BoundaryCondition.hpp"
#include "../geo/VoxelGeometry.hpp"

void TreeCallback(Fl_Widget *w, void *data);

class TreeView : public Fl_Tree
{
public:
  TreeView(int X, int Y, int W, int H);
};

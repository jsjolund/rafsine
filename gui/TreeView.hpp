#pragma once

#include <FL/Fl.H>
#include <FL/Fl_Tree.H>
#include <iostream>

#include "../sim/BoundaryCondition.hpp"
#include "../sim/KernelData.hpp"
#include "../geo/VoxelGeometry.hpp"
#include "DataTable.hpp"

#define UNIT_CONVERTER "Unit Conversion"
#define PHYSICAL_CONSTANTS "Physical Constants"
#define USER_CONSTANTS "User Constants"
#define GEOMETRY "Geometry"
#define BOUNDARY_CONDITIONS "Boundary Conditions"

void TreeCallback(Fl_Widget *w, void *data);
class TreeView : public Fl_Tree
{
private:
  std::unordered_map<Fl_Tree_Item *, VoxelGeometryObject *> treeObjectMap;
  KernelData *kernelData;

public:
  DataTable *table;

  TreeView(int X, int Y, int W, int H);
  void populate(KernelData *kernelData);
  void handleItem(Fl_Tree_Item *item, Fl_Tree_Reason reason);
};

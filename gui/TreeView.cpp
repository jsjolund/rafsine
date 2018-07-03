#include "TreeView.hpp"

void TreeView::populate(KernelData *kernelData_)
{
  kernelData = kernelData_;
  clear();
  treeObjectMap.clear();
  add(UNIT_CONVERTER);
  add(PHYSICAL_CONSTANTS);
  add(USER_CONSTANTS);

  std::stringstream ss;
  std::string bcs = GEOMETRY;
  for (int i = 0; i < kernelData->geo->size(); i++)
  {
    // For each group of quads and boxes
    VoxelGeometryGroup *group = kernelData->geo->at(i);
    ss.str("");
    ss << bcs << "/" << group->name;
    string groupPath = ss.str();
    Fl_Tree_Item *item = add(groupPath.c_str());
    treeObjectMap[item] = group;

    for (int j = 0; j < group->objs->size(); j++)
    {
      // For each quad/box in the group
      VoxelGeometryObject *obj = group->objs->at(j);
      ss.str("");

      if (dynamic_cast<VoxelGeometryBox *>(obj))
      {
        ss << groupPath << "/Box:" << obj->name;
        string boxGroupPath = ss.str();
        VoxelGeometryBox *box = dynamic_cast<VoxelGeometryBox *>(obj);
        Fl_Tree_Item *item = add(boxGroupPath.c_str());
        treeObjectMap[item] = box;

        for (int k = 0; k < box->quads.size(); k++)
        {
          // For each quad in the box
          VoxelGeometryQuad *quad = box->quads.at(k);
          ss.str("");
          ss << boxGroupPath << "/" << quad->name;
          Fl_Tree_Item *item = add(ss.str().c_str());
          treeObjectMap[item] = quad;
        }
      }
      else if (dynamic_cast<VoxelGeometryQuad *>(obj))
      {
        ss << groupPath << "/Quad:" << obj->name;
        Fl_Tree_Item *item = add(ss.str().c_str());
        treeObjectMap[item] = dynamic_cast<VoxelGeometryQuad *>(obj);
      }
    }
  }
  ss.str("");
  ss << "/" << bcs;
  close(ss.str().c_str());
}

void TreeView::handleItem(Fl_Tree_Item *item, Fl_Tree_Reason reason)
{
  table->clear();
  if (string(item->label()) == UNIT_CONVERTER)
  {
    std::cout << "handle unit converter" << std::endl;
    table->showUnitConverter(kernelData->uc);
  }
  else if (string(item->label()) == PHYSICAL_CONSTANTS)
  {
    std::cout << "handle physical constant" << std::endl;
    table->showSimConstants(kernelData->sc);
  }
  else if (string(item->label()) == USER_CONSTANTS)
  {
    std::cout << "handle user constant" << std::endl;
    table->showUserConstants(kernelData->c);
  }
  else if (string(item->label()) == GEOMETRY)
  {
    std::cout << "handle geo" << std::endl;
  }
  else if (string(item->label()) == BOUNDARY_CONDITIONS)
  {
    std::cout << "handle bc" << std::endl;
  }
  else if (treeObjectMap.find(item) != treeObjectMap.end())
  {
    VoxelGeometryObject *obj = treeObjectMap.at(item);
    // if (dynamic_cast<VoxelGeometryGroup *>(obj))
    // {
    //   std::cout << "handle GROUP " << obj->name << std::endl;
    //   VoxelGeometryGroup *group = dynamic_cast<VoxelGeometryGroup *>(obj);
    // }
    // else 
    if (dynamic_cast<VoxelGeometryBox *>(obj))
    {
      std::cout << "handle BOX " << obj->name << std::endl;
      VoxelGeometryBox *box = dynamic_cast<VoxelGeometryBox *>(obj);
    }
    else if (dynamic_cast<VoxelGeometryQuad *>(obj))
    {
      std::cout << "handle QUAD " << obj->name << std::endl;
      VoxelGeometryQuad *quad = dynamic_cast<VoxelGeometryQuad *>(obj);
      table->showVoxelGeometryQuad(quad);
    }
  }
}

void TreeCallback(Fl_Widget *w, void *data)
{
  TreeView *tree = static_cast<TreeView *>(w);
  Fl_Tree_Item *item = (Fl_Tree_Item *)tree->callback_item(); // get selected item
  tree->handleItem(item, tree->callback_reason());
  switch (tree->callback_reason())
  {
  case FL_TREE_REASON_SELECTED:
    std::cout << "selected" << std::endl;
    break;
  case FL_TREE_REASON_DESELECTED:
    std::cout << "deselected" << std::endl;
    break;
  case FL_TREE_REASON_OPENED:
    std::cout << "opened" << std::endl;
    break;
  case FL_TREE_REASON_CLOSED:
    std::cout << "closed" << std::endl;
    break;
  case FL_TREE_REASON_DRAGGED:
    std::cout << "dragged" << std::endl;
    break;
  case FL_TREE_REASON_NONE:
    std::cout << "none" << std::endl;
    break;
  }
}

TreeView::TreeView(int X, int Y, int W, int H)
    : Fl_Tree(X, Y, W, H, 0)
{
  showroot(0);
  callback(TreeCallback, (void *)1234);
}
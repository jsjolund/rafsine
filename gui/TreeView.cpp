#include "TreeView.hpp"

void TreeView::populate(KernelData *kernelData)
{
  clear();
  treeObjectMap.clear();
  add(UNIT_CONVERTER);
  add(PHYSICAL_CONSTANTS);
  add(USER_CONSTANTS);

  std::stringstream ss;
  std::string bcs = BOUNDARY_CONDITIONS;
  for (int i = 0; i < kernelData->geo->size(); i++)
  {
    // For each group of quads and boxes
    VoxelGeometryGroup *group = kernelData->geo->at(i);
    ss.str("");
    ss << bcs << "/" << group->name;
    string groupPath = ss.str();
    add(groupPath.c_str());

    for (int j = 0; j < group->objs->size(); j++)
    {
      // For each quad/box in the group
      VoxelGeometryObject *obj = group->objs->at(j);
      ss.str("");

      if (dynamic_cast<VoxelGeometryBox *>(obj))
      {
        ss << groupPath << "/Box:" << obj->name;
        string boxGroupPath = ss.str();
        Fl_Tree_Item *item = add(boxGroupPath.c_str());
        treeObjectMap[item] = obj;
        VoxelGeometryBox *box = dynamic_cast<VoxelGeometryBox *>(obj);
        for (int k = 0; k < box->quads.size(); k++)
        {
          VoxelGeometryObject quad = box->quads.at(k);
          ss.str("");
          ss << boxGroupPath << "/"<< quad.name;
          Fl_Tree_Item *item = add(ss.str().c_str());
          treeObjectMap[item] = obj;
        }
      }
      else if (dynamic_cast<VoxelGeometryQuad *>(obj))
      {
        ss << groupPath << "/Quad:" << obj->name;
        Fl_Tree_Item *item = add(ss.str().c_str());
        treeObjectMap[item] = obj;
      }
    }
  }
  ss.str("");
  ss << "/" << bcs;
  close(ss.str().c_str());
}

void TreeView::handleItem(Fl_Tree_Item *item, Fl_Tree_Reason reason)
{

  if (treeObjectMap.find(item) != treeObjectMap.end())
  {
    VoxelGeometryObject *obj = treeObjectMap.at(item);
    std::cout << "handle " << obj->name << std::endl;
  }
}

void TreeCallback(Fl_Widget *w, void *data)
{
  TreeView *tree = (TreeView *)w;
  Fl_Tree_Item *item = (Fl_Tree_Item *)tree->callback_item(); // get selected item
  tree->handleItem(item, tree->callback_reason());
  std::cout << "item=" << item->label() << std::endl;
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
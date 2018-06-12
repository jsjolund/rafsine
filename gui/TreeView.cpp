#include "TreeView.hpp"

void TreeCallback(Fl_Widget *w, void *data)
{
  Fl_Tree *tree = (Fl_Tree *)w;
  Fl_Tree_Item *item = (Fl_Tree_Item *)tree->callback_item(); // get selected item
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

TreeView::TreeView(int X, int Y, int W, int H, const char *L) : Fl_Tree(X, Y, W, H, L)
{
  st_tree::tree<BoundaryCondition> bcs;

  add("Boundary Conditions/Homer");
  close("/Boundary Conditions");
  add("Constants");
  add("User Variables");
  callback(TreeCallback, (void *)1234);
}
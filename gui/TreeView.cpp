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

TreeView::TreeView(int X, int Y, int W, int H) : Fl_Tree(X, Y, W, H, 0)
{
  showroot(0);
  add("Unit Conversion");
  add("Physical Constants");
  add("User Constants");

  add("Boundary Conditions/CRAC01");
  add("Boundary Conditions/CRAC01/Box:Unnamed");
  add("Boundary Conditions/CRAC01/Quad:Inlet");
  add("Boundary Conditions/CRAC01/Quad:Exhaust");
  add("Boundary Conditions/CRAC02");
  add("Boundary Conditions/CRAC02/Box:Unnamed");
  add("Boundary Conditions/CRAC02/Quad:Inlet");
  add("Boundary Conditions/CRAC02/Quad:Exhaust");
  close("/Boundary Conditions");
  callback(TreeCallback, (void *)1234);
}
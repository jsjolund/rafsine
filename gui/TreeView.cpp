#include "TreeView.hpp"

void TreeCallback(Fl_Widget *w, void *data)
{
  Fl_Tree *tree = (Fl_Tree *)w;
  Fl_Tree_Item *item = (Fl_Tree_Item *)tree->callback_item(); // get selected item
  fprintf(stderr, "item='%s'\n", item->label());
  switch (tree->callback_reason())
  {
  case FL_TREE_REASON_SELECTED:
    std::cout << "selected" << std::endl;
    break;
  case FL_TREE_REASON_DESELECTED:
    break;
  case FL_TREE_REASON_OPENED:
    break;
  case FL_TREE_REASON_CLOSED:
    break;
  }
}

TreeView::TreeView(int X, int Y, int W, int H, const char *L) : Fl_Tree(X, Y, W, H, L)
{
  add("Boundary Conditions/Homer");
  add("Boundary Conditions/Marge");
  add("Boundary Conditions/Bart");
  add("Boundary Conditions/Lisa");
  close("/Boundary Conditions");
  add("Constants");
  add("User Variables");
  callback(TreeCallback, (void *)1234);
}
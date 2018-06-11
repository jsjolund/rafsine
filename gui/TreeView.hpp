#pragma once

#include <FL/Fl.H>
#include <FL/Fl_Tree.H>

void TreeCallback(Fl_Widget *w, void *data);

class TreeView : public Fl_Tree
{
  public:
    TreeView(int X, int Y, int W, int H, const char *L = 0);
};

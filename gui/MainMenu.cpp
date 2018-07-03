#include "MainMenu.hpp"

void Change_CB(Fl_Widget *w, void *)
{
  Fl_Menu_Bar *menu = (Fl_Menu_Bar *)w;
  Fl_Menu_Item *p;
  // Change submenu name
  p = (Fl_Menu_Item *)menu->find_item("Edit/Submenu");
  if (p)
    p->label("New Submenu Name");
  // Change item name
  p = (Fl_Menu_Item *)menu->find_item("Edit/New Submenu Name/Aaa");
  if (p)
    p->label("New Aaa Name");
}

void Quit_CB(Fl_Widget *, void *)
{
  exit(0);
}

MainMenu::MainMenu(int X, int Y, int W, int H) : Fl_Menu_Bar(X, Y, W, H, "")
{
  add("File/Quit", FL_CTRL + 'q', Quit_CB);
  add("Edit/Change", FL_CTRL + 'c', Change_CB);
  add("Edit/Submenu/Aaa");
  add("Edit/Submenu/Bbb");
}
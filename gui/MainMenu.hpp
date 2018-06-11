#pragma once

#include <FL/Fl.H>
#include <FL/Fl_Tree.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Menu_Bar.H>

void Change_CB(Fl_Widget *w, void *);
void Quit_CB(Fl_Widget *, void *);

class MainMenu : public Fl_Menu_Bar
{
  public:
    MainMenu(int X, int Y, int W, int H);
};
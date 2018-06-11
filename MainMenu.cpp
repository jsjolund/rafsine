#pragma once

#include <FL/Fl.H>
#include <FL/Fl_Tree.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Box.H>

class MainMenuBar : public Fl_Menu_Bar
{
  public:
    MainMenuBar(int X, int Y, int W, int H) : Fl_Menu_Bar( X,  Y,  W,  H, "");
    {
        orig_h = H;
        last_y = 0;
        min_h = 10;
        align(FL_ALIGN_CENTER | FL_ALIGN_INSIDE);
        labelfont(FL_COURIER);
        labelsize(H);
        visible_focus(0);
        box(FL_UP_BOX);
        HandleDrag(0);
    }
};
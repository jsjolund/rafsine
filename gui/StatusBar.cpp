#include "StatusBar.hpp"

StatusBar::StatusBar(int X, int Y, int W, int H) : Fl_Box(X, Y, W, H, 0)
{
  box(FL_UP_FRAME);
}
#pragma once

#include <FL/Fl.H>
#include <FL/Fl_Tree.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Box.H>

class ResizerBarHoriz : public Fl_Box
{
    int orig_h;
    int last_y; 
    int min_h; // min height for widget above us
    void HandleDrag(int diff);

  public:
    ResizerBarHoriz(int X, int Y, int W, int H) : Fl_Box(X, Y, W, H, "")
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
    inline void SetMinHeight(int val) { min_h = val; }
    inline int GetMinHeight() const { return min_h; }
    int handle(int e);
    void resize(int X, int Y, int W, int H);
};

class ResizerBarVert : public Fl_Box
{
    int orig_w;
    int last_x;
    int min_w; // min height for widget above us
    void HandleDrag(int diff);

  public:
    ResizerBarVert(int X, int Y, int W, int H) : Fl_Box(X, Y, W, H, "")
    {
        orig_w = W;
        last_x = 0;
        min_w = 10;
        align(FL_ALIGN_CENTER | FL_ALIGN_INSIDE);
        labelfont(FL_COURIER);
        labelsize(W);
        visible_focus(0);
        box(FL_UP_BOX);
        HandleDrag(0);
    }

    inline void SetMinWidth(int val) { min_w = val; }
    inline int GetMinWidth() const { return min_w; }
    int handle(int e);
    void resize(int X, int Y, int W, int H);
};
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
    ResizerBarHoriz(int X, int Y, int W, int H);
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
    ResizerBarVert(int X, int Y, int W, int H);
    inline void SetMinWidth(int val) { min_w = val; }
    inline int GetMinWidth() const { return min_w; }
    int handle(int e);
    void resize(int X, int Y, int W, int H);
};
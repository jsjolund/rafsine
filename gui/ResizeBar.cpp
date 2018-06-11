#include "ResizeBar.hpp"

ResizerBarHoriz::ResizerBarHoriz(int X, int Y, int W, int H) : Fl_Box(X, Y, W, H, "")
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
void ResizerBarHoriz::HandleDrag(int diff)
{
    Fl_Double_Window *grp = (Fl_Double_Window *)parent();
    int top = y();
    int bot = y() + h();
    // First pass: find widget directly above us with common edge
    //    Possibly clamp 'diff' if widget would get too small..
    //
    for (int t = 0; t < grp->children(); t++)
    {
        Fl_Widget *w = grp->child(t);
        if ((w->y() + w->h()) == top)
        { // found widget directly above?
            if ((w->h() + diff) < min_h)
                diff = w->h() - min_h;                        // clamp
            w->resize(w->x(), w->y(), w->w(), w->h() + diff); // change height
            // break;                                            // done with first pass
        }
    }
    // Second pass: find widgets below us, move based on clamped diff
    for (int t = 0; t < grp->children(); t++)
    {
        Fl_Widget *w = grp->child(t);
        if (w->y() == bot)                                           // found widget below us?
            w->resize(w->x(), w->y() + diff, w->w(), w->h() - diff); // change position
    }
    // Change our position last
    resize(x(), y() + diff, w(), h());
    grp->init_sizes();
    grp->redraw();
}

int ResizerBarHoriz::handle(int e)
{
    int ret = 0;
    int this_y = Fl::event_y_root();
    switch (e)
    {
    case FL_FOCUS:
        ret = 1;
        break;
    case FL_ENTER:
        ret = 1;
        fl_cursor(FL_CURSOR_NS);
        break;
    case FL_LEAVE:
        ret = 1;
        fl_cursor(FL_CURSOR_DEFAULT);
        break;
    case FL_PUSH:
        ret = 1;
        last_y = this_y;
        break;
    case FL_DRAG:
        HandleDrag(this_y - last_y);
        last_y = this_y;
        ret = 1;
        break;
    default:
        break;
    }
    return (Fl_Box::handle(e) | ret);
}
void ResizerBarHoriz::resize(int X, int Y, int W, int H)
{
    Fl_Box::resize(X, Y, W, orig_h); // height of resizer stays constant size
}
/////////////////////////////////////////////
ResizerBarVert::ResizerBarVert(int X, int Y, int W, int H) : Fl_Box(X, Y, W, H, "")
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
void ResizerBarVert::HandleDrag(int diff)
{
    Fl_Double_Window *grp = (Fl_Double_Window *)parent();
    int left = x();
    int right = x() + w();
    // First pass: find widget directly above us with common edge
    //    Possibly clamp 'diff' if widget would get too small..
    //
    for (int t = 0; t < grp->children(); t++)
    {
        Fl_Widget *w = grp->child(t);
        if ((w->x() + w->w()) == left)
        { // found widget directly above?
            if ((w->w() + diff) < min_w)
                diff = w->w() - min_w;                        // clamp
            w->resize(w->x(), w->y(), w->w() + diff, w->h()); // change height
            // break;                                            // done with first pass
        }
    }
    // Second pass: find widgets below us, move based on clamped diff
    for (int t = 0; t < grp->children(); t++)
    {
        Fl_Widget *w = grp->child(t);
        if (w->x() == right)                                         // found widget below us?
            w->resize(w->x() + diff, w->y(), w->w() - diff, w->h()); // change position
    }
    // Change our position last
    resize(x() + diff, y(), w(), h());
    grp->init_sizes();
    grp->redraw();
}

int ResizerBarVert::handle(int e)
{
    int ret = 0;
    int this_x = Fl::event_x_root();
    switch (e)
    {
    case FL_FOCUS:
        ret = 1;
        break;
    case FL_ENTER:
        ret = 1;
        fl_cursor(FL_CURSOR_WE);
        break;
    case FL_LEAVE:
        ret = 1;
        fl_cursor(FL_CURSOR_DEFAULT);
        break;
    case FL_PUSH:
        ret = 1;
        last_x = this_x;
        break;
    case FL_DRAG:
        HandleDrag(this_x - last_x);
        last_x = this_x;
        ret = 1;
        break;
    default:
        break;
    }
    return (Fl_Box::handle(e) | ret);
}
void ResizerBarVert::resize(int X, int Y, int W, int H)
{
    Fl_Box::resize(X, Y, orig_w, H); // height of resizer stays constant size
}
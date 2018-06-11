#include "DataTable.hpp"

void DataTable::resize(int X, int Y, int W, int H)
{
  Fl_Scroll::resize(X, Y, W, H);
  int col1w = W - col0w;
  for (int r = 0; r < ROWS; r++)
  {
    Fl_Float_Input *in = (Fl_Float_Input*) w[r][1];
    in->resize(in->x(),in->y(),col1w,in->h());
  }
}

DataTable::DataTable(int X, int Y, int W, int H, const char *L)
    : Fl_Scroll(X, Y, W, H, L)
{
  static const char *header[COLS] = {"Property", "Value"};
  int col1w = W - col0w;
  int xx = X, yy = Y;
  tile = new Fl_Tile(X, Y, W, cellh * ROWS);
  // Create widgets
  for (int r = 0; r < ROWS; r++)
  {
    for (int c = 0; c < COLS; c++)
    {
      if (r == 0 && c == 0)
      {
        Fl_Box *box = new Fl_Box(xx, yy, col0w, cellh, header[c]);
        box->box(FL_UP_FRAME);
        w[r][c] = (void *)box;
      }
      else if (r == 0 && c == 1)
      {
        Fl_Box *box = new Fl_Box(xx, yy, col1w, cellh, header[c]);
        box->box(FL_UP_FRAME);
        w[r][c] = (void *)box;
      }
      else if (c == 0)
      {
        Fl_Input *in = new Fl_Input(xx, yy, col0w, cellh);
        in->box(FL_BORDER_BOX);
        in->value("");
        w[r][c] = (void *)in;
      }
      else
      {
        Fl_Float_Input *in = new Fl_Float_Input(xx, yy, col1w, cellh);
        in->box(FL_BORDER_BOX);
        in->value("0.00");
        w[r][c] = (void *)in;
      }
      xx += col0w;
    }
    xx = X;
    yy += cellh;
  }
  tile->end();
  end();
}
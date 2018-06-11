#pragma once

#include <FL/Fl_Tile.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Float_Input.H>
#include <FL/Fl_Scroll.H>

#define COLS 2
#define ROWS 20

class DataTable : public Fl_Scroll
{
private:
  void *w[ROWS][COLS];
  int cellh = 25;
  int col0w = 80;
  Fl_Tile *tile;

public:
  void resize(int X, int Y, int W, int H);
  DataTable(int X, int Y, int W, int H, const char *L = 0);
};
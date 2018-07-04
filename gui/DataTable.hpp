#pragma once

#include <FL/Fl_Tile.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Float_Input.H>
#include <FL/Fl_Scroll.H>

#include <vector>
#include <iostream>
#include <map>
#include <string>

#include "../sim/SimConstants.hpp"
#include "../sim/UnitConverter.hpp"
#include "../geo/VoxelGeometry.hpp"

class RealPointerInputBox : public Fl_Float_Input
{
private:
  real *ptr;

public:
  RealPointerInputBox(real *ptr, int X, int Y, int W, int H, const char *L);
  int handle(int event) override;
};

class LuaInputBox : public Fl_Input
{
private:
  std::string *ptr;

public:
  LuaInputBox(std::string *ptr, int X, int Y, int W, int H, const char *L);
  int handle(int event) override;
};

class DataTable : public Fl_Scroll
{
private:
  std::vector<Fl_Widget *> widgets;
  const int cellh;
  const int col0w;
  int rows;
  int cols;
  Fl_Tile *tile;

  void setTableHeaders(const char *header[2], int numRows);
  void showFloatTable(const char *header[2], tsl::ordered_map<const char *, real *> *cmap);
  void showStringTable(const char *header[2], tsl::ordered_map<string *, string *> *cmap);

public:
  void clear();
  void showVoxelGeometryQuad(VoxelGeometryQuad *quad);
  void showSimConstants(SimConstants *constants);
  void showUserConstants(UserConstants *uc);
  void showUnitConverter(UnitConverter *uc);
  void resize(int X, int Y, int W, int H) override;
  DataTable(int X, int Y, int W, int H, const char *L = 0);
};
#pragma once

#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Menu_Bar.H>

#include <iostream>

#include <osg/Quat>

#include "ResizeBar.hpp"
#include "GLWindow.hpp"
#include "TreeView.hpp"
#include "DataTable.hpp"
#include "MainMenu.hpp"
#include "../geo/VoxelMesh.hpp"

class MainWindow : public Fl_Double_Window
{
private:
  GLWindow *mygl;
  const int resize_h = 5;
  const int resize_w = 5;
  const int menu_h = 24;
  TreeView *tree;
  DataTable *table;
  ResizerBarHoriz *hbar;
  ResizerBarVert *vbar;
  MainMenu *menu;

public:
  void setVoxelMesh(VoxelMesh *mesh);
  void resize(int X, int Y, int W, int H);
  MainWindow(int W, int H, const char *L = 0);
};

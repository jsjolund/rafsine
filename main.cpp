#include <osgViewer/Viewer>
#include <osgViewer/CompositeViewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/TrackballManipulator>
#include <osgDB/ReadFile>

#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Menu_Bar.H>

#include <iostream>

#include "gui/MainWindow.hpp"
#include "geo/VoxelGeometry.hpp"
#include "geo/Voxel.hpp"
#include "geo/VoxelMesh.hpp"

#include "ext/st_tree/include/st_tree.h"

void idle_cb()
{
  Fl::redraw();
}

VoxelGeometry createGeometry()
{
  UnitConverter uc(6.95, 15, 1.0, 0.1, 1, 0, 0);
  real mx = 6.95;
  real my = 6.4;
  real mz = 3.1;

  VoxelGeometry vox(uc.m_to_lu(mx) + 1, uc.m_to_lu(my) + 1, uc.m_to_lu(mz) + 1, &uc);
  vox.addWallXmin();
  vox.addWallYmin();
  vox.addWallZmin();
  vox.addWallXmax();
  vox.addWallYmax();
  vox.addWallZmax();

  DomainGeometryBox box("test", vec3<real>(1, 1, 0), vec3<real>(3, 3, 2));
  vox.addSolidBox(&box);

  vox.saveToFile("test.vox");
  return vox;
}

int main(int argc, char **argv)
{
  MainWindow mainWindow(1280, 720, "LUA LBM GPU Leeds 2013");
  mainWindow.resizable(&mainWindow);

  VoxelGeometry vox = createGeometry();

  VoxelMesh mesh(*(vox.data));
  mesh.buildMesh();

  mainWindow.setVoxelMesh(&mesh);

  mainWindow.show();

  Fl::set_idle(idle_cb);

  return Fl::run();
}
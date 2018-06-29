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
#include "sim/SimConstants.hpp"

#include "ext/st_tree/include/st_tree.h"

void idle_cb()
{
  Fl::redraw();
}

VoxelGeometry createGeometry()
{
  UnitConverter uc;
  // reference length in meters
  uc.ref_L_phys = 6.95;
  // reference length in number of nodes
  uc.ref_L_lbm = 256;
  // reference speed in meter/second
  uc.ref_U_phys = 1.0;
  // reference speed in lattice units
  uc.ref_U_lbm = 0.1;
  // temperature conversion factor
  uc.C_Temp = 1;
  // reference temperature for Boussinesq in degres Celsius
  uc.T0_phys = 0;
  uc.T0_lbm = 0;

  SimConstants c(&uc);
  // Size of the lattice
  c.mx = 6.95;
  c.my = 6.4;
  c.mz = 3.1;
  // Kinematic viscosity of air
  c.nu = 1.568e-5;
  // Thermal diffusivity
  c.nuT = 1.0e-2;
  // Smagorinsky constant
  c.C = 0.02;
  // Thermal conductivity
  c.k = 2.624e-5;
  // Prandtl number of air
  c.Pr = 0.707;
  // Turbulent Prandtl number
  c.Pr_t = 0.9;
  // Gravity * thermal expansion
  c.gBetta = 9.82 * 3.32e-3;
  // Initial temperature
  c.Tinit = 30;
  // Reference temperature
  c.Tref = c.Tinit;

  VoxelGeometry vox(c.nx(), c.ny(), c.nz(), &uc);
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
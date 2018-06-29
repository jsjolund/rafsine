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
#include "sim/KernelData.hpp"

#include "ext/ordered-map/tsl/ordered_map.h"

void idle_cb()
{
  Fl::redraw();
}

KernelData createGeometry()
{
  UserConstants c;
  c["cracX"] = "0.510";
  c["cracY"] = "1.225";
  c["cracZ"] = "2.55";
  c["cracOutletY"] = "1.00";
  c["cracOutletZoffset"] = "0.1";
  c["cracOutletZ"] = "1.875 - cracOutletZoffset";

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
  // reference temperature for Boussinesq in degrees Celsius
  uc.T0_phys = 0;
  uc.T0_lbm = 0;

  SimConstants sc(&uc);
  // Size of the lattice
  sc.mx = 6.95;
  sc.my = 6.4;
  sc.mz = 3.1;
  // Kinematic viscosity of air
  sc.nu = 1.568e-5;
  // Thermal diffusivity
  sc.nuT = 1.0e-2;
  // Smagorinsky constant
  sc.C = 0.02;
  // Thermal conductivity
  sc.k = 2.624e-5;
  // Prandtl number of air
  sc.Pr = 0.707;
  // Turbulent Prandtl number
  sc.Pr_t = 0.9;
  // Gravity * thermal expansion
  sc.gBetta = 9.82 * 3.32e-3;
  // Initial temperature
  sc.Tinit = 30;
  // Reference temperature
  sc.Tref = sc.Tinit;

  VoxelGeometry vox(sc.nx(), sc.ny(), sc.nz(), &uc);
  std::vector<VoxelGeometryObject> wallQuads;
  wallQuads.push_back(vox.addWallXmin());
  wallQuads.push_back(vox.addWallYmin());
  wallQuads.push_back(vox.addWallZmin());
  wallQuads.push_back(vox.addWallXmax());
  wallQuads.push_back(vox.addWallYmax());
  wallQuads.push_back(vox.addWallZmax());

  VoxelGeometryBox box("test", vec3<real>(1, 1, 0), vec3<real>(3, 3, 2));
  std::vector<VoxelGeometryObject> boxQuads;
  vox.addSolidBox(&box, &boxQuads, &uc);

  KernelData data(&uc, &sc, &c, &vox);

  vox.saveToFile("test.vox");
  return data;
}

int main(int argc, char **argv)
{
  MainWindow mainWindow(1280, 720, "LUA LBM GPU Leeds 2013");
  mainWindow.resizable(&mainWindow);

  KernelData kernelData = createGeometry();

  VoxelMesh mesh(*(kernelData.vox->data));
  mesh.buildMesh();

  mainWindow.setVoxelMesh(&mesh);

  mainWindow.show();

  Fl::set_idle(idle_cb);

  return Fl::run();
}
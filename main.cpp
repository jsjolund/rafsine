#include <osgViewer/Viewer>
#include <osgViewer/CompositeViewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/TrackballManipulator>
#include <osgDB/ReadFile>

#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Menu_Bar.H>

//Lua is used to load the settings
extern "C"
{
#include <lua5.1/lua.hpp>
#include <lua5.1/lauxlib.h>
#include <lua5.1/lualib.h>
}
#undef luaL_dostring
#define luaL_dostring(L, s) \
  (luaL_loadstring(L, s) || lua_pcall(L, 0, LUA_MULTRET, 0))

#include <iostream>

#include "gui/MainWindow.hpp"
#include "geo/VoxelGeometry.hpp"
#include "geo/Voxel.hpp"
#include "geo/VoxelMesh.hpp"
#include "sim/SimConstants.hpp"
#include "sim/KernelData.hpp"

#include "ext/osgCompute/include/osgCuda/Computation"
#include "ext/osgCompute/include/osgCuda/Buffer"
#include "ext/osgCompute/include/osgCuda/Texture"
#include "ext/osgCompute/include/osgCudaStats/Stats"
#include "ext/osgCompute/include/osgCudaInit/Init"

void idle_cb()
{
  Fl::redraw();
}

int main(int argc, char **argv)
{
  //   lua_State *lua = lua_open();
  //   // luaL_openlibs(lua);
  //   luaopen_math(lua);
  //   string code = "a = 0.5\n b = 1.0 + 1.5 + math.sin(3.14) + a\n return b";
  //   int ret = luaL_dostring(lua, code.c_str());
  //   if (ret != 0)
  //   {
  //     printf("Error occurs when calling luaL_dofile() Hint Machine 0x%x\n", ret);
  //     printf("Error: %s\n", lua_tostring(lua, -1));
  //   }
  //   else
  //     printf("\nDOFILE SUCCESS\n");
  //   lua_getglobal(lua, "a");
  //   double a = (double)lua_tonumber(lua, -1);
  //   cout << "a = " << a << endl;
  // lua_getglobal(lua, "b");
  //   double b = (double)lua_tonumber(lua, -1);
  //   cout << "b = " << b << endl;

  //   int top = lua_gettop(lua);
  //   cout << "stack top is " << top << endl;
  //   lua_pushstring(lua, "derp2");
  //   top = lua_gettop(lua);
  //   cout << "stack top is " << top << endl;
  //   const char *test = lua_tostring(lua, -1);
  //   cout << test << endl;
  //   double testDouble = lua_tonumber(lua, -2);
  //   cout << testDouble + 1 << endl;
  //   lua_close(lua);
  //   return 0;

  // CUDA stream priorities. Simulation has highest priority, rendering lowest.
  cudaStream_t simStream;
  cudaStream_t renderStream;
  int priority_high, priority_low;
  cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
  cudaStreamCreateWithPriority(&simStream, cudaStreamNonBlocking, priority_high);
  cudaStreamCreateWithPriority(&renderStream, cudaStreamNonBlocking, priority_low);

  UnitConverter uc;
  // reference length in meters
  uc.ref_L_phys = 6.95;
  // reference length in number of nodes
  uc.ref_L_lbm = 256;
  // reference speed in meter/second
  uc.ref_U_phys = 1.0;
  // reference speed in lattice units
  uc.ref_U_lbm = 0.03;
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

  UserConstants c;
  c["cracX"] = "0.510";
  c["cracY"] = "1.225";
  c["cracZ"] = "2.55";
  c["cracOutletY"] = "1.00";
  c["cracOutletZoffset"] = "0.1";
  c["cracOutletZ"] = "1.875 - cracOutletZoffset";

  VoxelGeometry vox(sc.nx(), sc.ny(), sc.nz(), &uc);

  VoxelGeometryGroup wallQuads("Walls");
  VoxelGeometryQuad xmin = vox.addWallXmin();
  wallQuads.objs->push_back(&xmin);
  VoxelGeometryQuad ymin = vox.addWallYmin();
  wallQuads.objs->push_back(&ymin);
  VoxelGeometryQuad zmin = vox.addWallZmin();
  wallQuads.objs->push_back(&zmin);
  VoxelGeometryQuad xmax = vox.addWallXmax();
  wallQuads.objs->push_back(&xmax);
  VoxelGeometryQuad ymax = vox.addWallYmax();
  wallQuads.objs->push_back(&ymax);
  VoxelGeometryQuad zmax = vox.addWallZmax();
  wallQuads.objs->push_back(&zmax);

  vec3<std::string> testPoint("mx", "my", "mz");
  std::cout << testPoint << std::endl;

  VoxelGeometryGroup cracGeo("CRAC01");
  VoxelGeometryBox box("TestBox", vec3<real>(1, 2, 0), vec3<real>(3, 4, 2));
  vox.addSolidBox(&box, &uc);
  cracGeo.objs->push_back(&box);
  VoxelGeometryQuad quad("TestQuad",
                         NodeMode::Enum::OVERWRITE,
                         vec3<real>(1.5, 2, 0.5),
                         vec3<real>(1.2, 0, 0),
                         vec3<real>(0, 0, 1.2),
                         vec3<int>(0, 1, 0),
                         VoxelType::Enum::INLET_CONSTANT,
                         10,
                         vec3<int>(0, 1, 0));
  vox.addQuadBC(&quad, &uc);
  cracGeo.objs->push_back(&quad);

  vox.saveToFile("test.vox");

  KernelData kernelData(&uc, &sc, &c, &vox);
  kernelData.geo->push_back(&wallQuads);
  kernelData.geo->push_back(&cracGeo);

  VoxelMesh mesh(*(kernelData.vox->data));
  mesh.buildMesh();

  MainWindow mainWindow(1280, 720, "LUA LBM GPU Leeds 2013");
  mainWindow.setCudaRenderStream(renderStream);
  mainWindow.resizable(&mainWindow);
  mainWindow.setKernelData(&kernelData);
  mainWindow.setVoxelMesh(&mesh);
  mainWindow.show();


  Fl::set_idle(idle_cb);

  return Fl::run();
}
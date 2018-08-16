#include <QApplication>
#include <QMainWindow>
#include <QDesktopWidget>

#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "CFDWidget.hpp"
#include "VoxelGeometry.hpp"

#include "LuaContext.hpp"

// class VoxelGeometryTest
// {
// public:
//   VoxelGeometryTest() {}
//   void addQuadBC(
//       real originX, real originY, real originZ,
//       real dir1X, real dir1Y, real dir1Z,
//       real dir2X, real dir2Y, real dir2Z,
//       real normalX, real normalY, real normalZ,
//       real velocityX, real velocityY, real velocityZ,
//       std::string typeBC,
//       std::string temperatureType,
//       real temperature,
//       std::string mode,
//       std::string name)
//   {
//     std::cout << "added quad " << name << " mode " << mode << std::endl;
//   }
//   void addSolidBox(
//       real minX, real minY, real minZ,
//       real maxX, real maxY, real maxZ,
//       real temperature,
//       std::string name)
//   {
//     std::cout << "added vox " << name << std::endl;
//   }
// };

int main(int, char **)
{
  LuaContext lua;
  std::shared_ptr<UnitConverter> uc = std::make_shared<UnitConverter>();
  lua.writeVariable("ucAdapter", uc);

  lua.registerFunction("round", &UnitConverter::round);
  lua.registerFunction("set", &UnitConverter::set);
  lua.registerFunction("m_to_lu", (int (UnitConverter::*)(real))(&UnitConverter::m_to_lu));
  lua.registerFunction("m_to_LUA", (int (UnitConverter::*)(real))(&UnitConverter::m_to_LUA));
  lua.registerFunction("ms_to_lu", &UnitConverter::ms_to_lu);
  lua.registerFunction("Q_to_Ulu", &UnitConverter::Q_to_Ulu);
  lua.registerFunction("Nu_to_lu", &UnitConverter::Nu_to_lu);
  lua.registerFunction("Nu_to_tau", &UnitConverter::Nu_to_tau);
  lua.registerFunction("N_to_s", &UnitConverter::N_to_s);
  lua.registerFunction("s_to_N", &UnitConverter::s_to_N);
  lua.registerFunction("Temp_to_lu", &UnitConverter::Temp_to_lu);
  lua.registerFunction("gBetta_to_lu", &UnitConverter::gBetta_to_lu);

  std::ifstream script = std::ifstream{"lua/settings.lua"};
  lua.executeCode(script);
  script.close();

  float nx = lua.readVariable<float>("nx");
  float ny = lua.readVariable<float>("ny");
  float nz = lua.readVariable<float>("nz");

  std::shared_ptr<VoxelGeometry> vox = std::make_shared<VoxelGeometry>(nx, ny, nz, uc);
  lua.writeVariable("voxGeoAdapter", vox);
  lua.registerFunction("addWallXmin", &VoxelGeometry::addWallXmin);
  lua.registerFunction("addWallYmin", &VoxelGeometry::addWallYmin);
  lua.registerFunction("addWallZmin", &VoxelGeometry::addWallZmin);
  lua.registerFunction("addWallXmax", &VoxelGeometry::addWallXmax);
  lua.registerFunction("addWallYmax", &VoxelGeometry::addWallYmax);
  lua.registerFunction("addWallZmax", &VoxelGeometry::addWallZmax);
  lua.registerFunction("addQuadBC", &VoxelGeometry::createAddQuadBC);
  lua.registerFunction("addSolidBox", &VoxelGeometry::createAddSolidBox);

  script = std::ifstream{"lua/buildGeometry.lua"};
  lua.executeCode(script);
  script.close();

  return 0;

  // QApplication qapp(argc, argv);

  // QMainWindow window;
  // CFDWidget *widget = new CFDWidget(1, 1, &window);
  // window.setCentralWidget(widget);
  // window.show();
  // window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.3);
  // widget->setFocus();

  // return qapp.exec();
}
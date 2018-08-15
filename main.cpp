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

class VoxelGeometryTest
{
public:
  VoxelGeometryTest() {}

  void addWallXmin() { std::cout << "added wall xmin" << std::endl; }
  void addWallYmin() { std::cout << "added wall ymin" << std::endl; }
  void addWallZmin() { std::cout << "added wall zmin" << std::endl; }
  void addWallXmax() { std::cout << "added wall xmax" << std::endl; }
  void addWallYmax() { std::cout << "added wall ymax" << std::endl; }
  void addWallZmax() { std::cout << "added wall zmax" << std::endl; }
  void addQuadBC(
      real originX, real originY, real originZ,
      real dir1X, real dir1Y, real dir1Z,
      real dir2X, real dir2Y, real dir2Z,
      real normalX, real normalY, real normalZ,
      real velocityX, real velocityY, real velocityZ,
      std::string typeBC,
      std::string temperatureType,
      real temperature,
      std::string mode,
      std::string name)
  {
    std::cout << "added quad " << name << " mode " << mode << std::endl;
  }
  void addSolidBox(
      real minX, real minY, real minZ,
      real maxX, real maxY, real maxZ,
      real temperature,
      std::string name)
  {
    std::cout << "added vox " << name << std::endl;
  }
};

int main(int argc, char **argv)
{
  LuaContext lua;
  lua.registerFunction("addWallXmin", &VoxelGeometry::addWallXmin);
  lua.registerFunction("addWallYmin", &VoxelGeometry::addWallYmin);
  lua.registerFunction("addWallZmin", &VoxelGeometry::addWallZmin);
  lua.registerFunction("addWallXmax", &VoxelGeometry::addWallXmax);
  lua.registerFunction("addWallYmax", &VoxelGeometry::addWallYmax);
  lua.registerFunction("addWallZmax", &VoxelGeometry::addWallZmax);
  lua.registerFunction("addQuadBC", &VoxelGeometryTest::addQuadBC);
  lua.registerFunction("addSolidBox", &VoxelGeometryTest::addSolidBox);

  // lua.writeVariable("message", "hello");
  // lua.executeCode("adapter:increment(message,2);");
  // std::cout << lua.readVariable<VoxelGeometryTest>("adapter").value << std::endl;

  lua.writeVariable("adapter", VoxelGeometryTest{});
  std::ifstream script = std::ifstream{"lua/buildGeometry.lua"};
  lua.executeCode(script);

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
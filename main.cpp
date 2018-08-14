#include <QApplication>
#include <QMainWindow>
#include <QDesktopWidget>

#include <iostream>
#include <unistd.h>
#include <stdio.h>

#include "CFDWidget.hpp"
#include "VoxelGeometry.hpp"
#include "Voxel.hpp"
#include "VoxelMesh.hpp"
#include "SimConstants.hpp"
#include "KernelData.hpp"

#include "LuaContext.hpp"

class Object
{
public:
  Object() : value(10) {}

  void increment()
  {
    std::cout << "incrementing" << std::endl;
    value++;
  }

  int value;
};

int main(int argc, char **argv)
{
  LuaContext lua;
  lua.registerFunction("increment", &Object::increment);
  lua.writeVariable("obj", Object{});
  lua.executeCode("obj:increment();");
  std::cout << lua.readVariable<Object>("obj").value << std::endl;

  lua.executeCode("obj:increment();");

  QApplication qapp(argc, argv);

  QMainWindow window;
  CFDWidget *widget = new CFDWidget(1, 1, &window);
  window.setCentralWidget(widget);
  window.show();
  window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.3);
  widget->setFocus();

  return qapp.exec();
}
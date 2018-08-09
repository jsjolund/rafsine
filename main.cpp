#include <QApplication>
#include <QMainWindow>
#include <QDesktopWidget>

#include <iostream>
#include <unistd.h>
#include <stdio.h>

#include "gui/CFDWidget.hpp"
#include "geo/VoxelGeometry.hpp"
#include "geo/Voxel.hpp"
#include "geo/VoxelMesh.hpp"
#include "sim/SimConstants.hpp"
#include "sim/KernelData.hpp"

int main(int argc, char **argv)
{
  QApplication qapp(argc, argv);

  QMainWindow window;
  CFDWidget *widget = new CFDWidget(1, 1, &window);
  window.setCentralWidget(widget);
  window.show();
  window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.3);
  widget->setFocus();

  return qapp.exec();
}
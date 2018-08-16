#include <QApplication>
#include <QMainWindow>
#include <QDesktopWidget>

#include <iostream>
#include <unistd.h>
#include <stdio.h>

#include "CFDWidget.hpp"

int main(int argc, char **argv)
{
  QApplication qapp(argc, argv);

  QMainWindow window;
  CFDWidget *widget = new CFDWidget(1, 1, &window);
  window.setCentralWidget(widget);
  window.show();
  window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.5);
  widget->setFocus();

  return qapp.exec();
}
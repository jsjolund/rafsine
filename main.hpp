#pragma once

#include <QApplication>
#include <QMainWindow>
#include <QOpenGLWidget>
#include <QMouseEvent>
#include <QObject>
#include <QEvent>
#include <QKeyEvent>
#include <QWheelEvent>
#include <QDesktopWidget>
#include <QScreen>
#include <QtGlobal>
#include <QWindow>

#include <osg/ref_ptr>
#include <osgViewer/GraphicsWindow>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osg/Camera>
#include <osg/ShapeDrawable>
#include <osg/StateSet>
#include <osg/Material>
#include <osgGA/EventQueue>
#include <osgGA/TrackballManipulator>

#include <iostream>
#include <stdio.h>

#include "ext/osgCompute/include/osgCudaInit/Init"
#include "ext/osgCompute/include/osgCudaStats/Stats"

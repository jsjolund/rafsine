// #pragma once

// #include <QMainWindow>
// #include <QOpenGLWidget>
// #include <QMouseEvent>
// #include <QObject>
// #include <QEvent>
// #include <QKeyEvent>
// #include <QWheelEvent>
// #include <QDesktopWidget>
// #include <QScreen>
// #include <QtGlobal>
// #include <QWindow>

// #include <osg/ref_ptr>
// #include <osgViewer/GraphicsWindow>
// #include <osgViewer/Viewer>
// #include <osgViewer/ViewerEventHandlers>
// #include <osg/Camera>
// #include <osg/ShapeDrawable>
// #include <osg/StateSet>
// #include <osg/Material>
// #include <osgGA/EventQueue>
// #include <osgGA/TrackballManipulator>

// #include <iostream>
// #include <stdio.h>

// #include "../ext/osgCompute/include/osgCudaInit/Init"
// #include "../ext/osgCompute/include/osgCudaStats/Stats"

// #include "CFDScene.hpp"
// #include "PickHandler.hpp"

// class QtOSGWidget : public QOpenGLWidget
// {
// private:
//   CFDScene *scene_;

//   osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> _mGraphicsWindow;
//   osg::ref_ptr<osgViewer::Viewer> _mViewer;
//   qreal m_scaleX, m_scaleY;

//   inline osgGA::EventQueue *getEventQueue() const
//   {
//     osgGA::EventQueue *eventQueue = _mGraphicsWindow->getEventQueue();
//     return eventQueue;
//   }

// protected:
//   virtual void paintGL();
//   virtual void resizeGL(int width, int height);
//   virtual void initializeGL();
//   virtual void mouseMoveEvent(QMouseEvent *event);
//   virtual void mousePressEvent(QMouseEvent *event);
//   void keyPressEvent(QKeyEvent *e);
//   virtual void mouseReleaseEvent(QMouseEvent *event);
//   virtual void wheelEvent(QWheelEvent *event);
//   virtual bool event(QEvent *event);


// public:
//   QtOSGWidget(qreal scaleX, qreal scaleY, QWidget *parent = 0);
//   virtual ~QtOSGWidget();
//   void setScale(qreal X, qreal Y);
//   inline CFDScene* getScene() { return scene_; }
//   void setVoxelMesh(VoxelMesh *voxmesh);
// };
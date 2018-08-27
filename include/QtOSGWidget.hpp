
#pragma once

#include <QOpenGLWidget>
#include <QMouseEvent>
#include <QApplication>

#include <osg/ref_ptr>
#include <osgViewer/GraphicsWindow>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/EventQueue>
#include <osgGA/TrackballManipulator>

class QtOSGWidget : public QOpenGLWidget
{
private:
  inline osgGA::EventQueue *getEventQueue() const
  {
    osgGA::EventQueue *eventQueue = m_gfxWindow->getEventQueue();
    return eventQueue;
  }

protected:
  osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> m_gfxWindow;
  osg::ref_ptr<osgViewer::Viewer> m_viewer;
  qreal m_scaleX, m_scaleY;

  virtual void initializeGL() = 0;
  virtual void paintGL() = 0;

  virtual void resizeGL(int width, int height);

  unsigned int getMouseButton(QMouseEvent *event);
  virtual void mouseDoubleClickEvent(QMouseEvent *event);
  virtual void mouseMoveEvent(QMouseEvent *event);
  virtual void mousePressEvent(QMouseEvent *event);
  virtual void mouseReleaseEvent(QMouseEvent *event);
  virtual void wheelEvent(QWheelEvent *event);

  void keyPressEvent(QKeyEvent *event);

  virtual bool event(QEvent *event);

public:
  QtOSGWidget(qreal scaleX, qreal scaleY, QWidget *parent = 0);
  virtual ~QtOSGWidget();

  void setScale(qreal X, qreal Y);

  inline osg::ref_ptr<osgViewer::Viewer> getViewer() { return m_viewer; }
};
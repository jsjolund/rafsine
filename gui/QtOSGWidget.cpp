#include "QtOSGWidget.hpp"

QtOSGWidget::QtOSGWidget(qreal scaleX, qreal scaleY, QWidget *parent)
    : QOpenGLWidget(parent),
      m_gfxWindow(new osgViewer::GraphicsWindowEmbedded(this->x(),
                                                        this->y(),
                                                        this->width(),
                                                        this->height())),
      m_viewer(new osgViewer::Viewer),
      m_scaleX(scaleX),
      m_scaleY(scaleY)
{
  osg::Camera *camera = new osg::Camera;
  camera->setViewport(0, 0, this->width(), this->height());
  camera->setClearColor(osg::Vec4(0.9f, 0.9f, 0.9f, 1.f));
  float aspectRatio = static_cast<float>(this->width()) / static_cast<float>(this->height());
  camera->setProjectionMatrixAsPerspective(30.f, aspectRatio, 1.f, 1000.f);
  camera->setGraphicsContext(m_gfxWindow);

  m_viewer->setCamera(camera);
  osgGA::TrackballManipulator *manipulator = new osgGA::TrackballManipulator;
  manipulator->setAllowThrow(false);
  this->setMouseTracking(true);
  m_viewer->setCameraManipulator(manipulator);
  m_viewer->addEventHandler(new osgViewer::StatsHandler);
  m_viewer->setRunFrameScheme(osgViewer::ViewerBase::FrameScheme::CONTINUOUS);
  m_viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);
  m_viewer->realize();
}

QtOSGWidget::~QtOSGWidget() {}

void QtOSGWidget::setScale(qreal X, qreal Y)
{
  m_scaleX = X;
  m_scaleY = Y;
  this->resizeGL(this->width(), this->height());
}

void QtOSGWidget::resizeGL(int width, int height)
{
  this->getEventQueue()->windowResize(this->x() * m_scaleX, this->y() * m_scaleY, width * m_scaleX, height * m_scaleY);
  m_gfxWindow->resized(this->x() * m_scaleX, this->y() * m_scaleY, width * m_scaleX, height * m_scaleY);
  osg::Camera *camera = m_viewer->getCamera();
  camera->setViewport(0, 0, this->width() * m_scaleX, this->height() * m_scaleY);
}

void QtOSGWidget::mouseMoveEvent(QMouseEvent *event)
{
  this->getEventQueue()->mouseMotion(event->x() * m_scaleX, event->y() * m_scaleY);
}

void QtOSGWidget::mousePressEvent(QMouseEvent *event)
{
  setFocus();
  unsigned int button = 0;
  switch (event->button())
  {
  case Qt::LeftButton:
    button = 1;
    break;
  case Qt::MiddleButton:
    button = 2;
    break;
  case Qt::RightButton:
    button = 3;
    break;
  default:
    break;
  }
  this->getEventQueue()->mouseButtonPress(event->x() * m_scaleX, event->y() * m_scaleY, button);
}

void QtOSGWidget::keyPressEvent(QKeyEvent *event)
{
  typedef osgGA::GUIEventAdapter::KeySymbol OSGKey;

  int qtKey = event->key();
  OSGKey osgKey;

  switch (qtKey)
  {
  case Qt::Key_Escape:
    osgKey = OSGKey::KEY_Escape;
    break;
  case Qt::Key_Tab:
    osgKey = OSGKey::KEY_Tab;
    break;
  case Qt::Key_Backspace:
    osgKey = OSGKey::KEY_BackSpace;
    break;
  case Qt::Key_Return:
    osgKey = OSGKey::KEY_Return;
    break;
  case Qt::Key_Enter:
    osgKey = OSGKey::KEY_Return;
    break;
  case Qt::Key_Insert:
    osgKey = OSGKey::KEY_Insert;
    break;
  case Qt::Key_Delete:
    osgKey = OSGKey::KEY_Delete;
    break;
  case Qt::Key_Pause:
    osgKey = OSGKey::KEY_Pause;
    break;
  case Qt::Key_Print:
    osgKey = OSGKey::KEY_Print;
    break;
  case Qt::Key_SysReq:
    osgKey = OSGKey::KEY_Sys_Req;
    break;
  case Qt::Key_Clear:
    osgKey = OSGKey::KEY_Clear;
    break;
  case Qt::Key_Home:
    osgKey = OSGKey::KEY_Home;
    break;
  case Qt::Key_End:
    osgKey = OSGKey::KEY_End;
    break;
  case Qt::Key_Left:
    osgKey = OSGKey::KEY_Left;
    break;
  case Qt::Key_Up:
    osgKey = OSGKey::KEY_Up;
    break;
  case Qt::Key_Right:
    osgKey = OSGKey::KEY_Right;
    break;
  case Qt::Key_Down:
    osgKey = OSGKey::KEY_Down;
    break;
  case Qt::Key_PageUp:
    osgKey = OSGKey::KEY_Page_Up;
    break;
  case Qt::Key_PageDown:
    osgKey = OSGKey::KEY_Page_Down;
    break;
  case Qt::Key_Shift:
    osgKey = OSGKey::KEY_Shift_L;
    break;
  case Qt::Key_Control:
    osgKey = OSGKey::KEY_Control_L;
    break;
  case Qt::Key_Alt:
    osgKey = OSGKey::KEY_Alt_L;
    break;
  case Qt::Key_AltGr:
    osgKey = OSGKey::KEY_Meta_R;
    break;
  case Qt::Key_CapsLock:
    osgKey = OSGKey::KEY_Caps_Lock;
    break;
  case Qt::Key_ScrollLock:
    osgKey = OSGKey::KEY_Shift_Lock;
    break;
  case Qt::Key_F1:
    osgKey = OSGKey::KEY_F1;
    break;
  case Qt::Key_F2:
    osgKey = OSGKey::KEY_F2;
    break;
  case Qt::Key_F3:
    osgKey = OSGKey::KEY_F3;
    break;
  case Qt::Key_F4:
    osgKey = OSGKey::KEY_F4;
    break;
  case Qt::Key_F5:
    osgKey = OSGKey::KEY_F5;
    break;
  case Qt::Key_F6:
    osgKey = OSGKey::KEY_F6;
    break;
  case Qt::Key_F7:
    osgKey = OSGKey::KEY_F7;
    break;
  case Qt::Key_F8:
    osgKey = OSGKey::KEY_F8;
    break;
  case Qt::Key_F9:
    osgKey = OSGKey::KEY_F9;
    break;
  case Qt::Key_F10:
    osgKey = OSGKey::KEY_F10;
    break;
  case Qt::Key_F11:
    osgKey = OSGKey::KEY_F11;
    break;
  case Qt::Key_F12:
    osgKey = OSGKey::KEY_F12;
    break;
  case Qt::Key_Super_L:
    osgKey = OSGKey::KEY_Super_L;
    break;
  case Qt::Key_Super_R:
    osgKey = OSGKey::KEY_Super_R;
    break;
  case Qt::Key_Menu:
    osgKey = OSGKey::KEY_Menu;
    break;
  case Qt::Key_Hyper_L:
    osgKey = OSGKey::KEY_Hyper_L;
    break;
  case Qt::Key_Hyper_R:
    osgKey = OSGKey::KEY_Hyper_R;
    break;
  default:
    const char *asciiCode = event->text().toLatin1().data();
    m_gfxWindow->getEventQueue()->keyPress(osgGA::GUIEventAdapter::KeySymbol(*asciiCode));
    return;
  }
  m_gfxWindow->getEventQueue()->keyPress(osgKey);
}

void QtOSGWidget::mouseReleaseEvent(QMouseEvent *event)
{
  unsigned int button = 0;
  switch (event->button())
  {
  case Qt::LeftButton:
    button = 1;
    break;
  case Qt::MiddleButton:
    button = 2;
    break;
  case Qt::RightButton:
    button = 3;
    break;
  default:
    break;
  }
  this->getEventQueue()->mouseButtonRelease(event->x() * m_scaleX, event->y() * m_scaleY, button);
}

void QtOSGWidget::wheelEvent(QWheelEvent *event)
{
  int delta = event->delta();
  osgGA::GUIEventAdapter::ScrollingMotion motion = delta <= 0 ? osgGA::GUIEventAdapter::SCROLL_UP : osgGA::GUIEventAdapter::SCROLL_DOWN;
  this->getEventQueue()->mouseScroll(motion);
}

bool QtOSGWidget::event(QEvent *event)
{
  bool handled = QOpenGLWidget::event(event);
  this->update();
  return handled;
}

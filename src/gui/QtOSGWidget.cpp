#include "QtOSGWidget.hpp"

QtOSGWidget::QtOSGWidget(qreal scaleX, qreal scaleY, QWidget* parent)
    : QOpenGLWidget(parent),
      m_prevRefTime(0),
      m_gfxWindow(new osgViewer::GraphicsWindowEmbedded(this->x(),
                                                        this->y(),
                                                        this->width(),
                                                        this->height())),
      m_viewer(new osgViewer::Viewer),
      m_statsHandler(new osgViewer::StatsHandler),
      m_scaleX(scaleX),
      m_scaleY(scaleY) {
  osg::ref_ptr<osg::Camera> camera = new osg::Camera;
  camera->setViewport(0, 0, this->width(), this->height());
  camera->setClearColor(BACKGROUND_COLOR);
  float aspectRatio =
      static_cast<float>(this->width()) / static_cast<float>(this->height());
  camera->setProjectionMatrixAsPerspective(30.0, aspectRatio, 1.0, 1000.0);
  camera->setGraphicsContext(m_gfxWindow);
  camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
  setMouseTracking(true);

  m_cameraManipulator = new MyOrbitManipulator(camera);
  m_cameraManipulator->setAllowThrow(false);
  m_viewer->setCameraManipulator(m_cameraManipulator);

  m_viewer->setCamera(camera);
  
  m_viewer->addEventHandler(m_statsHandler);

  m_viewer->setRunFrameScheme(osgViewer::ViewerBase::FrameScheme::ON_DEMAND);
  m_viewer->setThreadingModel(osgViewer::Viewer::AutomaticSelection);
  m_viewer->setReleaseContextAtEndOfFrameHint(false);
  m_viewer->setKeyEventSetsDone(0);
}

QtOSGWidget::~QtOSGWidget() {}

void QtOSGWidget::paintGL() {
  double refTime = m_viewer->getViewerFrameStamp()->getReferenceTime();
  double deltaFrameTime = refTime - m_prevRefTime;
  render(deltaFrameTime);
  m_prevRefTime = refTime;
  m_viewer->frame();
}

void QtOSGWidget::homeCamera() {
  m_cameraManipulator->home(0);
  double dst = m_cameraManipulator->getDistance();
  m_cameraManipulator->setDistance(dst / 2);
  osg::Quat rotation = osg::Quat(osg::PI / 4, osg::Vec3d(1, 0, 0)) *
                       osg::Quat(-osg::PI / 4, osg::Vec3d(0, 0, 1));
  m_cameraManipulator->setRotation(rotation);
}

void QtOSGWidget::setOrthographicCamera(bool state) {
  float w = static_cast<float>(this->width());
  float h = static_cast<float>(this->height());
  m_cameraManipulator->setOrthographicCamera(state, w, h);
}

void QtOSGWidget::setScale(qreal X, qreal Y) {
  m_scaleX = X;
  m_scaleY = Y;
  this->resizeGL(this->width(), this->height());
}

void QtOSGWidget::resizeGL(int width, int height) {
  this->getEventQueue()->windowResize(this->x() * m_scaleX,
                                      this->y() * m_scaleY, width * m_scaleX,
                                      height * m_scaleY);
  m_gfxWindow->resized(this->x() * m_scaleX, this->y() * m_scaleY,
                       width * m_scaleX, height * m_scaleY);
  osg::Camera* camera = m_viewer->getCamera();
  camera->setViewport(0, 0, this->width() * m_scaleX,
                      this->height() * m_scaleY);
  m_statsHandler->getCamera()->setViewport(0, 0, this->width() * m_scaleX,
                                           this->height() * m_scaleY);
}

void QtOSGWidget::mouseMoveEvent(QMouseEvent* event) {
  this->getEventQueue()->mouseMotion(event->x() * m_scaleX,
                                     event->y() * m_scaleY);
}

unsigned int QtOSGWidget::getMouseButton(QMouseEvent* event) {
  unsigned int button = 0;
  switch (event->button()) {
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
  return button;
}

void QtOSGWidget::mouseDoubleClickEvent(QMouseEvent* event) {
  setFocus();
  unsigned int button = getMouseButton(event);
  this->getEventQueue()->mouseDoubleButtonPress(event->x() * m_scaleX,
                                                event->y() * m_scaleY, button);
}

void QtOSGWidget::mousePressEvent(QMouseEvent* event) {
  setFocus();
  unsigned int button = getMouseButton(event);
  this->getEventQueue()->mouseButtonPress(event->x() * m_scaleX,
                                          event->y() * m_scaleY, button);
}

void QtOSGWidget::mouseReleaseEvent(QMouseEvent* event) {
  unsigned int button = getMouseButton(event);
  this->getEventQueue()->mouseButtonRelease(event->x() * m_scaleX,
                                            event->y() * m_scaleY, button);
}

void QtOSGWidget::wheelEvent(QWheelEvent* event) {
  int delta = event->delta();
  osgGA::GUIEventAdapter::ScrollingMotion motion =
      delta <= 0 ? osgGA::GUIEventAdapter::SCROLL_UP
                 : osgGA::GUIEventAdapter::SCROLL_DOWN;
  this->getEventQueue()->mouseScroll(motion);
}

bool QtOSGWidget::event(QEvent* event) {
  bool handled = QOpenGLWidget::event(event);
  this->update();
  return handled;
}

static osgGA::GUIEventAdapter::KeySymbol getOsgKey(QKeyEvent* event) {
  typedef osgGA::GUIEventAdapter::KeySymbol osgKey;
  int qtKey = event->key();
  osgKey key;
  switch (qtKey) {
    case Qt::Key_Escape:
      key = osgKey::KEY_Escape;
      break;
    case Qt::Key_Tab:
      key = osgKey::KEY_Tab;
      break;
    case Qt::Key_Backspace:
      key = osgKey::KEY_BackSpace;
      break;
    case Qt::Key_Return:
      key = osgKey::KEY_Return;
      break;
    case Qt::Key_Enter:
      key = osgKey::KEY_Return;
      break;
    case Qt::Key_Insert:
      key = osgKey::KEY_Insert;
      break;
    case Qt::Key_Delete:
      key = osgKey::KEY_Delete;
      break;
    case Qt::Key_Pause:
      key = osgKey::KEY_Pause;
      break;
    case Qt::Key_Print:
      key = osgKey::KEY_Print;
      break;
    case Qt::Key_SysReq:
      key = osgKey::KEY_Sys_Req;
      break;
    case Qt::Key_Clear:
      key = osgKey::KEY_Clear;
      break;
    case Qt::Key_Home:
      key = osgKey::KEY_Home;
      break;
    case Qt::Key_End:
      key = osgKey::KEY_End;
      break;
    case Qt::Key_Left:
      key = osgKey::KEY_Left;
      break;
    case Qt::Key_Up:
      key = osgKey::KEY_Up;
      break;
    case Qt::Key_Right:
      key = osgKey::KEY_Right;
      break;
    case Qt::Key_Down:
      key = osgKey::KEY_Down;
      break;
    case Qt::Key_PageUp:
      key = osgKey::KEY_Page_Up;
      break;
    case Qt::Key_PageDown:
      key = osgKey::KEY_Page_Down;
      break;
    case Qt::Key_Shift:
      key = osgKey::KEY_Shift_L;
      break;
    case Qt::Key_Control:
      key = osgKey::KEY_Control_L;
      break;
    case Qt::Key_Alt:
      key = osgKey::KEY_Alt_L;
      break;
    case Qt::Key_AltGr:
      key = osgKey::KEY_Meta_R;
      break;
    case Qt::Key_CapsLock:
      key = osgKey::KEY_Caps_Lock;
      break;
    case Qt::Key_ScrollLock:
      key = osgKey::KEY_Shift_Lock;
      break;
    case Qt::Key_F1:
      key = osgKey::KEY_F1;
      break;
    case Qt::Key_F2:
      key = osgKey::KEY_F2;
      break;
    case Qt::Key_F3:
      key = osgKey::KEY_F3;
      break;
    case Qt::Key_F4:
      key = osgKey::KEY_F4;
      break;
    case Qt::Key_F5:
      key = osgKey::KEY_F5;
      break;
    case Qt::Key_F6:
      key = osgKey::KEY_F6;
      break;
    case Qt::Key_F7:
      key = osgKey::KEY_F7;
      break;
    case Qt::Key_F8:
      key = osgKey::KEY_F8;
      break;
    case Qt::Key_F9:
      key = osgKey::KEY_F9;
      break;
    case Qt::Key_F10:
      key = osgKey::KEY_F10;
      break;
    case Qt::Key_F11:
      key = osgKey::KEY_F11;
      break;
    case Qt::Key_F12:
      key = osgKey::KEY_F12;
      break;
    case Qt::Key_Super_L:
      key = osgKey::KEY_Super_L;
      break;
    case Qt::Key_Super_R:
      key = osgKey::KEY_Super_R;
      break;
    case Qt::Key_Menu:
      key = osgKey::KEY_Menu;
      break;
    case Qt::Key_Hyper_L:
      key = osgKey::KEY_Hyper_L;
      break;
    case Qt::Key_Hyper_R:
      key = osgKey::KEY_Hyper_R;
      break;
    default:
      const char* asciiCode = event->text().toLatin1().data();
      return osgGA::GUIEventAdapter::KeySymbol(*asciiCode);
  }
  return key;
}

void QtOSGWidget::keyPressEvent(QKeyEvent* event) {
  m_gfxWindow->getEventQueue()->keyPress(getOsgKey(event));
}

void QtOSGWidget::keyReleaseEvent(QKeyEvent* event) {
  m_gfxWindow->getEventQueue()->keyRelease(getOsgKey(event));
}

// #include "QtOSGWidget.hpp"

// QtOSGWidget::QtOSGWidget(qreal scaleX, qreal scaleY, QWidget *parent)
//     : QOpenGLWidget(parent),
//       _mGraphicsWindow(
//           new osgViewer::GraphicsWindowEmbedded(this->x(), this->y(),
//                                                 this->width(), this->height())),
//       _mViewer(new osgViewer::Viewer),
//       m_scaleX(scaleX),
//       m_scaleY(scaleY)
// {
//   // osg::Cylinder *cylinder = new osg::Cylinder(osg::Vec3(0.f, 0.f, 0.f), 0.25f, 0.5f);
//   // osg::ShapeDrawable *sd = new osg::ShapeDrawable(cylinder);
//   // sd->setColor(osg::Vec4(0.8f, 0.5f, 0.2f, 1.f));
//   // osg::Geode *geode = new osg::Geode;
//   // geode->addDrawable(sd);

//   // osg::Camera *camera = new osg::Camera;
//   // camera->setViewport(0, 0, this->width(), this->height());
//   // camera->setClearColor(osg::Vec4(0.9f, 0.9f, 1.f, 1.f));
//   // float aspectRatio = static_cast<float>(this->width()) / static_cast<float>(this->height());
//   // camera->setProjectionMatrixAsPerspective(30.f, aspectRatio, 1.f, 1000.f);
//   // camera->setGraphicsContext(_mGraphicsWindow);

//   // _mViewer->setCamera(camera);
//   // _mViewer->setSceneData(geode);
//   // osgGA::TrackballManipulator *manipulator = new osgGA::TrackballManipulator;
//   // manipulator->setAllowThrow(false);
//   // this->setMouseTracking(true);
//   // _mViewer->setCameraManipulator(manipulator);
//   // _mViewer->addEventHandler(new osgViewer::StatsHandler);
//   // _mViewer->addEventHandler(new osgCuda::StatsHandler);
//   // _mViewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);
//   // _mViewer->realize();
//   scene_ = new CFDScene();

//   osg::Camera *camera = new osg::Camera;
//   camera->setViewport(0, 0, this->width(), this->height());
//   float aspectRatio = static_cast<float>(this->width()) / static_cast<float>(this->height());
//   camera->setProjectionMatrixAsPerspective(30.f, aspectRatio, 1.f, 1000.f);
//   camera->setGraphicsContext(_mGraphicsWindow);
//   // camera->setDrawBuffer(GL_BACK);
//   // camera->setReadBuffer(GL_BACK);

//   _mViewer->setCamera(camera);

//   osgGA::TrackballManipulator *manipulator = new osgGA::TrackballManipulator;
//   manipulator->setAllowThrow(false);
//   this->setMouseTracking(true);
//   _mViewer->setCameraManipulator(manipulator);
//   _mViewer->addEventHandler(new osgViewer::StatsHandler);
//   // _mViewer->addEventHandler(new osgCuda::StatsHandler);
//   _mViewer->addEventHandler(new osgViewer::LODScaleHandler);
//   _mViewer->addEventHandler(new PickHandler());
//   _mViewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);
//   _mViewer->realize();

//   // osgCuda::setupOsgCudaAndViewer(*_mViewer);

//   _mViewer->setSceneData(scene_->getRoot());
// }

// QtOSGWidget::~QtOSGWidget() {}

// void QtOSGWidget::setScale(qreal X, qreal Y)
// {
//   m_scaleX = X;
//   m_scaleY = Y;
//   this->resizeGL(this->width(), this->height());
// }

// void QtOSGWidget::paintGL()
// {
//   _mViewer->frame();
// }

// void QtOSGWidget::resizeGL(int width, int height)
// {
//   this->getEventQueue()->windowResize(this->x() * m_scaleX, this->y() * m_scaleY, width * m_scaleX, height * m_scaleY);
//   _mGraphicsWindow->resized(this->x() * m_scaleX, this->y() * m_scaleY, width * m_scaleX, height * m_scaleY);
//   osg::Camera *camera = _mViewer->getCamera();
//   camera->setViewport(0, 0, this->width() * m_scaleX, this->height() * m_scaleY);
// }

// void QtOSGWidget::initializeGL()
// {
//   osg::Node *geode = dynamic_cast<osg::Group *>(_mViewer->getSceneData());

//   osg::StateSet *stateSet = geode->getOrCreateStateSet();
//   osg::Material *material = new osg::Material;
//   material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
//   stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
//   stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
// }

// void QtOSGWidget::mouseMoveEvent(QMouseEvent *event)
// {
//   this->getEventQueue()->mouseMotion(event->x() * m_scaleX, event->y() * m_scaleY);
// }

// void QtOSGWidget::mousePressEvent(QMouseEvent *event)
// {
//   unsigned int button = 0;
//   switch (event->button())
//   {
//   case Qt::LeftButton:
//     button = 1;
//     break;
//   case Qt::MiddleButton:
//     button = 2;
//     break;
//   case Qt::RightButton:
//     button = 3;
//     break;
//   default:
//     break;
//   }
//   this->getEventQueue()->mouseButtonPress(event->x() * m_scaleX, event->y() * m_scaleY, button);
//   setFocus();
// }

// void QtOSGWidget::keyPressEvent(QKeyEvent *e)
// {
//   std::cout << "key " << e->key() << std::endl;
//   const char *keyData = e->text().toLatin1().data();
//   _mGraphicsWindow->getEventQueue()->keyPress(osgGA::GUIEventAdapter::KeySymbol(*keyData));
// }

// void QtOSGWidget::mouseReleaseEvent(QMouseEvent *event)
// {
//   unsigned int button = 0;
//   switch (event->button())
//   {
//   case Qt::LeftButton:
//     button = 1;
//     break;
//   case Qt::MiddleButton:
//     button = 2;
//     break;
//   case Qt::RightButton:
//     button = 3;
//     break;
//   default:
//     break;
//   }
//   this->getEventQueue()->mouseButtonRelease(event->x() * m_scaleX, event->y() * m_scaleY, button);
// }

// void QtOSGWidget::wheelEvent(QWheelEvent *event)
// {
//   int delta = event->delta();
//   osgGA::GUIEventAdapter::ScrollingMotion motion = delta > 0 ? osgGA::GUIEventAdapter::SCROLL_UP : osgGA::GUIEventAdapter::SCROLL_DOWN;
//   this->getEventQueue()->mouseScroll(motion);
// }

// bool QtOSGWidget::event(QEvent *event)
// {
//   bool handled = QOpenGLWidget::event(event);
//   this->update();
//   return handled;
// }

// void QtOSGWidget::setVoxelMesh(VoxelMesh *voxmesh)
// {
//   scene_->setVoxelMesh(voxmesh);

//   float radius = voxmesh->getRadius() * 2;
//   osg::Vec3 eye(radius, radius, radius);
//   osg::Vec3 center = scene_->getCenter();
//   osg::Vec3 up(0.0, 0.0, 1.0);
//   _mViewer->getCameraManipulator()->setHomePosition(eye, center, up);
//   _mViewer->getCameraManipulator()->home(0);
//   std::cout << "set scene" << std::endl;
// }
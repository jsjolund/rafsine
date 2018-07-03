#include "GLWindow.hpp"

void GLWindow::setVoxelMesh(VoxelMesh *mesh)
{
  osg::Group *root = new osg::Group();
  osg::Geode *geode = new osg::Geode();
  osg::Geometry *polyGeom = new osg::Geometry();
  polyGeom->setVertexArray(mesh->vertices_);
  polyGeom->setColorArray(mesh->v_colors_);
  polyGeom->setNormalArray(mesh->normals_);
  polyGeom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
  polyGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
  int numVerts = mesh->vertices_->getNumElements();
  polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, numVerts));
  geode->addDrawable(polyGeom);
  // geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);
  // geode->getOrCreateStateSet()->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
  // geode->getOrCreateStateSet()->setMode(GL_TEXTURE_2D, osg::StateAttribute::OFF);
  root->addChild(geode);

  osg::Group *lightGroup = new osg::Group();
  osg::LightSource *lightSource = new osg::LightSource;
  osg::Light *light = new osg::Light;
  light->setAmbient(osg::Vec4(1.0, 1.0, 1.0, 1.0));
  light->setDiffuse(osg::Vec4(1.0, 1.0, 1.0, 1.0));
  light->setSpecular(osg::Vec4(1, 1, 1, 1));

  osg::Quat q = osg::Quat(osg::PI / 4, osg::Vec3d(1, 0, 0),
                          0, osg::Vec3d(0, 1, 0),
                          osg::PI / 4, osg::Vec3d(0, 0, 1));

  light->setDirection(q * osg::Vec3(1.0f, 0.0f, 0.0f));
  lightSource->setLight(light);
  lightGroup->addChild(lightSource);
  root->addChild(lightGroup);

  setSceneData(root);
}

void AdapterWidget::resize(int x, int y, int w, int h)
{
  _gw->getEventQueue()->windowResize(x, y, w, h);
  _gw->resized(x, y, w, h);

  Fl_Gl_Window::resize(x, y, w, h);
}

int AdapterWidget::handle(int event)
{
  int key = Fl::event_key();
  switch (event)
  {
  case FL_FOCUS:
    std::cout << "focus" << std::endl;
    return 1;
  case FL_PUSH:
    Fl::focus(this);
    std::cout << "push" << std::endl;
    _gw->getEventQueue()->mouseButtonPress(Fl::event_x(), Fl::event_y(), Fl::event_button());
    return 1;
  case FL_MOVE:
    return 1;
  case FL_DRAG:
    _gw->getEventQueue()->mouseMotion(Fl::event_x(), Fl::event_y());
    return 1;
  case FL_RELEASE:
    _gw->getEventQueue()->mouseButtonRelease(Fl::event_x(), Fl::event_y(), Fl::event_button());
    return 1;
  case FL_KEYDOWN:
    std::cout << "keydown" << std::endl;
    switch (key)
    {
    case FL_Insert:
      // plot->sliceXup();
      return 1;
    case FL_Delete:
      // plot->sliceXdown();
      return 1;
    case FL_Home:
      // plot->sliceYup();
      return 1;
    case FL_End:
      // plot->sliceYdown();
      return 1;
    case FL_Page_Up:
      // plot->sliceZup();
      return 1;
    case FL_Page_Down:
      // plot->sliceZdown();
      std::cout << "pgdn" << std::endl;
      return 1;
    default:
      _gw->getEventQueue()->keyPress((osgGA::GUIEventAdapter::KeySymbol)Fl::event_key());
      return 1;
    }
  case FL_KEYUP:
    std::cout << "keyup" << std::endl;
    _gw->getEventQueue()->keyRelease((osgGA::GUIEventAdapter::KeySymbol)Fl::event_key());
    return 1;
  default:
    // pass other events to the base class
    return Fl_Gl_Window::handle(event);
  }
}

GLWindow::GLWindow(int x, int y, int w, int h, const char *label)
    : AdapterWidget(x, y, w, h, label)
{
  getCamera()->setViewport(new osg::Viewport(0, 0, w, h));
  getCamera()->setProjectionMatrixAsPerspective(30.0f, static_cast<double>(w) / static_cast<double>(h), 1.0, 10000.0);
  getCamera()->setGraphicsContext(getGraphicsWindow());
  getCamera()->setDrawBuffer(GL_BACK);
  getCamera()->setReadBuffer(GL_BACK);
  setThreadingModel(osgViewer::Viewer::SingleThreaded);
  setCameraManipulator(new osgGA::TrackballManipulator);
  addEventHandler(new osgViewer::StatsHandler);
  addEventHandler(new PickHandler());
}
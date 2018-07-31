#include "GLWindow.hpp"

//increase slice x position
void GLWindow::sliceXup()
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if (slice_pos_.x < (long)vox_size_.x - 1)
      slice_pos_.x++;
    break;
  case DisplayMode::VOX_GEOMETRY:
    if (vox_min_.x < (long)vox_size_.x)
      vox_min_.x++;
    redrawVoxelMesh();
    break;
  }
}

//decrease slice x position
void GLWindow::sliceXdown()
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if (slice_pos_.x > 1)
      slice_pos_.x--;
    break;
  case DisplayMode::VOX_GEOMETRY:
    if (vox_min_.x >= 0)
      vox_min_.x--;
    redrawVoxelMesh();
    break;
  }
}

void GLWindow::sliceYup()
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if (slice_pos_.y < (long)vox_size_.y - 1)
      slice_pos_.y++;
    break;
  case DisplayMode::VOX_GEOMETRY:
    if (vox_min_.y < (long)vox_size_.y)
      vox_min_.y++;
    redrawVoxelMesh();
    break;
  }
}

void GLWindow::sliceYdown()
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if (slice_pos_.y > 1)
      slice_pos_.y--;
    break;
  case DisplayMode::VOX_GEOMETRY:
    if (vox_min_.y >= 0)
      vox_min_.y--;
    redrawVoxelMesh();
    break;
  }
}

void GLWindow::sliceZup()
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if (slice_pos_.z < (long)vox_size_.z - 1)
      slice_pos_.z++;
    break;
  case DisplayMode::VOX_GEOMETRY:
    if (vox_max_.z < (long)vox_size_.z)
      vox_max_.z++;
    redrawVoxelMesh();
    break;
  }
}

void GLWindow::sliceZdown()
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if (slice_pos_.z > 1)
      slice_pos_.z--;
    break;
  case DisplayMode::VOX_GEOMETRY:
    if (vox_max_.z >= 0)
      vox_max_.z--;
    redrawVoxelMesh();
    break;
  }
}

void GLWindow::setSliceXpos(int pos)
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if ((pos >= 0) && (pos < (long)vox_size_.x))
      slice_pos_.x = pos;
    break;
  case DisplayMode::VOX_GEOMETRY:
    vox_min_.x = pos;
    redrawVoxelMesh();
    break;
  }
}

void GLWindow::setSliceYpos(int pos)
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if ((pos >= 0) && (pos < (long)vox_size_.y))
      slice_pos_.y = pos;
    break;
  case DisplayMode::VOX_GEOMETRY:
    vox_min_.y = pos;
    redrawVoxelMesh();
    break;
  }
}

void GLWindow::setSliceZpos(int pos)
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if ((pos >= 0) && (pos < (long)vox_size_.z))
      slice_pos_.z = pos;
    break;
  case DisplayMode::VOX_GEOMETRY:
    vox_max_.z = pos;
    redrawVoxelMesh();
    break;
  }
}

void GLWindow::redrawVoxelMesh()
{
  voxmesh_->buildMesh(vox_min_.x, vox_max_.x,
                      vox_min_.y, vox_max_.y,
                      vox_min_.z, vox_max_.z);
  voxGeo->setVertexArray(voxmesh_->vertices_);
  voxGeo->setColorArray(voxmesh_->v_colors_);
  voxGeo->setNormalArray(voxmesh_->normals_);
  voxGeo->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
  voxGeo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
  if (voxGeo->getPrimitiveSetList().size() > 0)
    voxGeo->removePrimitiveSet(0);
  voxGeo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, voxmesh_->vertices_->getNumElements()));
}

void GLWindow::setVoxelMesh(VoxelMesh *mesh)
{
  voxmesh_ = mesh;
  vox_size_ = vec3<int>(voxmesh_->getSizeX(), voxmesh_->getSizeY(), voxmesh_->getSizeZ());
  slice_pos_ = vox_size_ / 2;
  vox_min_ = vec3<int>(-1, -1, -1);
  vox_max_ = vox_size_;
  redrawVoxelMesh();
  float radius = voxmesh_->getRadius() * 2;
  osg::Vec3 eye(radius, radius, radius);
  osg::Vec3 center(slice_pos_.x, slice_pos_.y, slice_pos_.z);
  osg::Vec3 up(0.0, 0.0, 1.0);
  getCameraManipulator()->setHomePosition(eye, center, up);
  getCameraManipulator()->home(0);
}

void AdapterWidget::resize(int x, int y, int w, int h)
{
  _gw->getEventQueue()->windowResize(x, y, w, h);
  _gw->resized(x, y, w, h);

  Fl_Gl_Window::resize(x, y, w, h);
}

int GLWindow::handle(int event)
{
  int key = Fl::event_key();
  switch (event)
  {
  case FL_FOCUS:
    // std::cout << "focus" << std::endl;
    return 1;
  case FL_PUSH:
    Fl::focus(this);
    // std::cout << "push" << std::endl;
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
    // std::cout << "keydown" << std::endl;
    switch (key)
    {
    case FL_Insert:
      sliceXup();
      return 1;
    case FL_Delete:
      sliceXdown();
      return 1;
    case FL_Home:
      sliceYup();
      return 1;
    case FL_End:
      sliceYdown();
      return 1;
    case FL_Page_Up:
      sliceZup();
      return 1;
    case FL_Page_Down:
      sliceZdown();
      return 1;
    default:
      _gw->getEventQueue()->keyPress((osgGA::GUIEventAdapter::KeySymbol)Fl::event_key());
      return 1;
    }
  case FL_KEYUP:
    // std::cout << "keyup" << std::endl;
    _gw->getEventQueue()->keyRelease((osgGA::GUIEventAdapter::KeySymbol)Fl::event_key());
    return 1;
  default:
    // pass other events to the base class
    return Fl_Gl_Window::handle(event);
  }
}

void GLWindow::draw()
{
  frame();

  if (!sliceX_) {
    sliceX_ = new SliceRender(renderStream_, vox_size_.y, vox_size_.z);
    sliceY_ = new SliceRender(renderStream_, vox_size_.x, vox_size_.z);
    sliceZ_ = new SliceRender(renderStream_, vox_size_.x, vox_size_.y);
    sliceC_ = new SliceRender(renderStream_, 128, 128);
  }

  sliceX_->compute(0, 100);
}

GLWindow::GLWindow(int x, int y, int w, int h, const char *label)
    : AdapterWidget(x, y, w, h, label),
      voxmesh_(NULL),
      displayMode_(DisplayMode::Enum::VOX_GEOMETRY),
      displayQuantity_(DisplayQuantity::Enum::TEMPERATURE),
      vox_size_(0, 0, 0),
      vox_max_(0, 0, 0),
      vox_min_(0, 0, 0),
      slice_pos_(0, 0, 0),
      sliceX_(NULL),
      sliceY_(NULL),
      sliceZ_(NULL),
      sliceC_(NULL)
{
  getCamera()->setViewport(new osg::Viewport(0, 0, w, h));
  getCamera()->setProjectionMatrixAsPerspective(30.0f, static_cast<double>(w) / static_cast<double>(h), 1.0, 10000.0);
  getCamera()->setGraphicsContext(getGraphicsWindow());
  getCamera()->setDrawBuffer(GL_BACK);
  getCamera()->setReadBuffer(GL_BACK);
  setThreadingModel(osgViewer::Viewer::AutomaticSelection);
  setCameraManipulator(new osgGA::TrackballManipulator);
  addEventHandler(new osgViewer::StatsHandler);
  addEventHandler(new osgViewer::LODScaleHandler);
  addEventHandler(new PickHandler());

  osg::Group *root = new osg::Group();
  osg::Geode *geode = new osg::Geode();
  voxGeo = new osg::Geometry();
  voxGeoTransform = new osg::PositionAttitudeTransform();

  geode->addDrawable(voxGeo);
  voxGeoTransform->addChild(geode);
  root->addChild(voxGeoTransform);

  voxGeoTransform->setPosition(osg::Vec3(0, 0, 0));

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
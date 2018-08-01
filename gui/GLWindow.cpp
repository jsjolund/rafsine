#include "GLWindow.hpp"

__global__ void SliceZRenderKernel(real *plot3D, int nx, int ny, int nz, real *plot2D, int slice_pos)
{
  int x, y;
  idx2d(x, y, nx);
  if ((x >= nx) || (y >= ny))
    return;
  //plot2D[x+nx*y] = plot3D[I3D(x, y, slice_pos, nx,ny,nz)];
  //gaussian blur
  int xp = (x == nx - 1) ? (x) : (x + 1);
  int xm = (x == 0) ? (x) : (x - 1);
  int yp = (y == ny - 1) ? (y) : (y + 1);
  int ym = (y == 0) ? (y) : (y - 1);
  plot2D[x + nx * y] =
      1 / 4.f * plot3D[I3D(x, y, slice_pos, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(xp, y, slice_pos, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(xm, y, slice_pos, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(x, yp, slice_pos, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(x, ym, slice_pos, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(xm, ym, slice_pos, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(xm, yp, slice_pos, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(xp, ym, slice_pos, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(xp, yp, slice_pos, nx, ny, nz)];
  //average over the height
  /*
  float average = 0;
  for(int z=0; z<nz; z++)
    average += plot3D[I3D(x, y, z, nx,ny,nz)];
  plot2D[x+nx*y] = average/nz;
  */
}

__global__ void SliceYRenderKernel(real *plot3D, int nx, int ny, int nz, real *plot2D, int slice_pos)
{
  int x, z;
  idx2d(x, z, nx);
  if ((x >= nx) || (z >= nz))
    return;
  //plot2D[x+nx*z] = plot3D[I3D(x, slice_pos, z, nx,ny,nz)];
  //gaussian blur
  int xp = (x == nx - 1) ? (x) : (x + 1);
  int xm = (x == 0) ? (x) : (x - 1);
  int zp = (z == nz - 1) ? (z) : (z + 1);
  int zm = (z == 0) ? (z) : (z - 1);
  plot2D[x + nx * z] =
      1 / 4.f * plot3D[I3D(x, slice_pos, z, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(xp, slice_pos, z, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(xm, slice_pos, z, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(x, slice_pos, zp, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(x, slice_pos, zm, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(xm, slice_pos, zm, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(xm, slice_pos, zp, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(xp, slice_pos, zm, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(xp, slice_pos, zp, nx, ny, nz)];
}

__global__ void SliceXRenderKernel(real *plot3D, int nx, int ny, int nz, real *plot2D, int slice_pos)
{
  int y, z;
  idx2d(y, z, ny);
  if ((y >= ny) || (z >= nz))
    return;
  //plot2D[y+ny*z] = plot3D[I3D(slice_pos, y, z, nx,ny,nz)];
  //gaussian blur
  int yp = (y == ny - 1) ? (y) : (y + 1);
  int ym = (y == 0) ? (y) : (y - 1);
  int zp = (z == nz - 1) ? (z) : (z + 1);
  int zm = (z == 0) ? (z) : (z - 1);
  plot2D[y + ny * z] =
      1 / 4.f * plot3D[I3D(slice_pos, y, z, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(slice_pos, yp, z, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(slice_pos, ym, z, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(slice_pos, y, zp, nx, ny, nz)] +
      1 / 8.f * plot3D[I3D(slice_pos, y, zm, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(slice_pos, ym, zm, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(slice_pos, ym, zp, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(slice_pos, yp, zm, nx, ny, nz)] +
      1 / 16.f * plot3D[I3D(slice_pos, yp, zp, nx, ny, nz)];
}

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
    case FL_F + 1:
      setDisplayMode(DisplayMode::SLICE);
      return 1;
    case FL_F + 2:
      setDisplayMode(DisplayMode::VOX_GEOMETRY);
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

void GLWindow::drawSliceX()
{
  dim3 block_size, grid_size;
  setDims(vox_size_.y * vox_size_.z, BLOCK_SIZE_DEFAULT, block_size, grid_size);
  SliceXRenderKernel<<<grid_size, block_size, 0, renderStream_>>>(
      gpu_ptr(), vox_size_.x, vox_size_.y, vox_size_.z, sliceX_->gpu_ptr(), slice_pos_.x);
  sliceX_->compute(min_, max_);
  sliceX_->transform->setPosition(osg::Vec3d(slice_pos_.x, 0, 0));
}

void GLWindow::drawSliceY()
{
  dim3 block_size, grid_size;
  setDims(vox_size_.x * vox_size_.z, BLOCK_SIZE_DEFAULT, block_size, grid_size);
  SliceYRenderKernel<<<grid_size, block_size, 0, renderStream_>>>(
      gpu_ptr(), vox_size_.x, vox_size_.y, vox_size_.z, sliceY_->gpu_ptr(), slice_pos_.y);
  sliceY_->compute(min_, max_);
  sliceY_->transform->setPosition(osg::Vec3d(0, slice_pos_.y, 0));
}

void GLWindow::drawSliceZ()
{
  dim3 block_size, grid_size;
  setDims(vox_size_.x * vox_size_.y, BLOCK_SIZE_DEFAULT, block_size, grid_size);
  SliceZRenderKernel<<<grid_size, block_size, 0, renderStream_>>>(
      gpu_ptr(), vox_size_.x, vox_size_.y, vox_size_.z, sliceZ_->gpu_ptr(), slice_pos_.z);
  sliceZ_->compute(min_, max_);
  sliceZ_->transform->setPosition(osg::Vec3d(0, 0, slice_pos_.z));
}

void GLWindow::draw()
{
  if (plot_d_.size() == 0)
  {
    plot_d_.resize(vox_size_.x * vox_size_.y * vox_size_.z, 20.0);

    sliceX_ = new SliceRender(renderStream_, vox_size_.y, vox_size_.z);
    sliceY_ = new SliceRender(renderStream_, vox_size_.x, vox_size_.z);
    sliceZ_ = new SliceRender(renderStream_, vox_size_.x, vox_size_.y);
    sliceC_ = new SliceRender(renderStream_, sizeC_, sizeC_);

    root_->addChild(sliceX_->transform);
    root_->addChild(sliceY_->transform);
    root_->addChild(sliceZ_->transform);

    sliceX_->transform->setAttitude(osg::Quat(osg::PI / 2, osg::Vec3d(0, 0, 1)));
    sliceY_->transform->setAttitude(osg::Quat(0, osg::Vec3d(0, 0, 1)));
    sliceZ_->transform->setAttitude(osg::Quat(-osg::PI / 2, osg::Vec3d(1, 0, 0)));
  }

  drawSliceX();
  drawSliceY();
  drawSliceZ();

  frame();
}

GLWindow::GLWindow(int x, int y, int w, int h, const char *label)
    : AdapterWidget(x, y, w, h, label),
      voxmesh_(NULL),
      displayMode_(DisplayMode::Enum::VOX_GEOMETRY),
      vox_size_(0, 0, 0),
      vox_max_(0, 0, 0),
      vox_min_(0, 0, 0),
      slice_pos_(0, 0, 0),
      sliceX_(NULL),
      sliceY_(NULL),
      sliceZ_(NULL),
      sliceC_(NULL),
      plot_d_(0),
      min_(0),
      max_(40),
      sizeC_(128),
      plot_c_(sizeC_ * sizeC_)
{
  getCamera()->setViewport(new osg::Viewport(0, 0, w, h));
  getCamera()->setProjectionMatrixAsPerspective(30.0f, static_cast<double>(w) / static_cast<double>(h), 1.0, 10000.0);
  getCamera()->setGraphicsContext(getGraphicsWindow());
  getCamera()->setDrawBuffer(GL_BACK);
  getCamera()->setReadBuffer(GL_BACK);
  setThreadingModel(osgViewer::Viewer::SingleThreaded);
  setCameraManipulator(new osgGA::TrackballManipulator);
  addEventHandler(new osgViewer::StatsHandler);
  addEventHandler(new osgViewer::LODScaleHandler);
  addEventHandler(new PickHandler());

  root_ = new osg::Group();
  osg::Geode *geode = new osg::Geode();
  voxGeo = new osg::Geometry();
  voxGeoTransform = new osg::PositionAttitudeTransform();

  geode->addDrawable(voxGeo);
  voxGeoTransform->addChild(geode);
  root_->addChild(voxGeoTransform);

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
  root_->addChild(lightGroup);

  setSceneData(root_);
}
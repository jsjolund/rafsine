#include "CFDScene.hpp"

// void CFDScene::frame(osg::Camera &camera)
// {
//   osg::Vec3d eye, center, up;
//   camera.getViewMatrixAsLookAt(eye, center, up);
//   up.normalize();
//   osg::Vec3d forward = center - eye;
//   forward.normalize();
//   forward = forward * 5;
//   osg::Vec3d down = up * (-1);
//   m_axesTransform->setPosition(eye + forward + down);
// }

void CFDScene::redrawVoxelMesh()
{
  m_voxMesh->buildMesh(*m_voxMin, *m_voxMax);
}

void CFDScene::setDisplayMode(DisplayMode::Enum mode)
{
  m_displayMode = mode;
  if (mode == DisplayMode::SLICE)
  {
    if (m_voxMesh)
      m_voxMesh->setNodeMask(0);
    if (m_sliceX)
      m_sliceX->setNodeMask(~0);
    if (m_sliceY)
      m_sliceY->setNodeMask(~0);
    if (m_sliceZ)
      m_sliceZ->setNodeMask(~0);
  }
  else
  {
    if (m_voxMesh)
      m_voxMesh->setNodeMask(~0);
    if (m_sliceX)
      m_sliceX->setNodeMask(0);
    if (m_sliceY)
      m_sliceY->setNodeMask(0);
    if (m_sliceZ)
      m_sliceZ->setNodeMask(0);
  }
}

void CFDScene::adjustDisplayColors()
{
  if (m_plot3d.size() == 0)
    return;
  thrust::device_vector<real>::iterator iter;
  iter = thrust::min_element(m_plot3d.begin(), m_plot3d.end());
  m_plotMin = *iter;
  iter = thrust::max_element(m_plot3d.begin(), m_plot3d.end());
  m_plotMax = *iter;
  std::cout << "Lattice, plot min: " << m_plotMin << "; plot max: " << m_plotMax << std::endl;
  m_sliceX->setMinMax(m_plotMin, m_plotMax);
  m_sliceY->setMinMax(m_plotMin, m_plotMax);
  m_sliceZ->setMinMax(m_plotMin, m_plotMax);
}

void CFDScene::setVoxelMesh(VoxelMesh *mesh)
{
  // CUDA stream priorities. Simulation has highest priority, rendering lowest.
  // This must be done in the thread which first runs a kernel
  cudaStream_t simStream = 0;
  cudaStream_t renderStream = 0;
  int priority_high, priority_low;
  cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
  cudaStreamCreateWithPriority(&simStream, cudaStreamNonBlocking, priority_high);
  // cudaStreamCreateWithPriority(&renderStream, cudaStreamDefault, priority_low);
  cudaStreamCreateWithPriority(&renderStream, cudaStreamNonBlocking, priority_low);

  // Clear the scene
  if (m_root->getNumChildren() > 0)
    m_root->removeChildren(0, m_root->getNumChildren());

  // Add voxel mesh to scene
  m_voxMesh = mesh;
  m_voxSize = new osg::Vec3i(m_voxMesh->getSizeX(), m_voxMesh->getSizeY(), m_voxMesh->getSizeZ());
  m_voxMin = new osg::Vec3i(-1, -1, -1);
  m_voxMax = new osg::Vec3i(*m_voxSize);
  m_root->addChild(m_voxMesh->getTransform());

  // Create a test 3D plot
  m_plot3d.erase(m_plot3d.begin(), m_plot3d.end());
  m_plot3d.reserve(m_voxMesh->getSizeX() * m_voxMesh->getSizeY() * m_voxMesh->getSizeZ());
  m_plot3d.resize(m_voxMesh->getSizeX() * m_voxMesh->getSizeY() * m_voxMesh->getSizeZ(), 0);
  thrust::counting_iterator<real> iter(0);
  thrust::copy(iter, iter + m_plot3d.size(), m_plot3d.begin());

  real *plot3dPtr = thrust::raw_pointer_cast(&(m_plot3d)[0]);

  // Add slice renderers to the scene
  m_slicePositions = new osg::Vec3i(*m_voxSize);
  *m_slicePositions = *m_slicePositions / 2;

  m_sliceX = new SliceRender(SliceRenderAxis::X_AXIS, m_voxSize->y(), m_voxSize->z(), plot3dPtr, *m_voxSize, renderStream);
  m_sliceX->getTransform()->setAttitude(osg::Quat(osg::PI / 2, osg::Vec3d(0, 0, 1)));
  m_sliceX->getTransform()->setPosition(osg::Vec3d(m_slicePositions->x(), 0, 0));
  m_root->addChild(m_sliceX->getTransform());

  m_sliceY = new SliceRender(SliceRenderAxis::Y_AXIS, m_voxSize->x(), m_voxSize->z(), plot3dPtr, *m_voxSize, renderStream);
  m_sliceY->getTransform()->setAttitude(osg::Quat(0, osg::Vec3d(0, 0, 1)));
  m_sliceY->getTransform()->setPosition(osg::Vec3d(0, m_slicePositions->y(), 0));
  m_root->addChild(m_sliceY->getTransform());

  m_sliceZ = new SliceRender(SliceRenderAxis::Z_AXIS, m_voxSize->x(), m_voxSize->y(), plot3dPtr, *m_voxSize, renderStream);
  m_sliceZ->getTransform()->setAttitude(osg::Quat(-osg::PI / 2, osg::Vec3d(1, 0, 0)));
  m_sliceZ->getTransform()->setPosition(osg::Vec3d(0, 0, m_slicePositions->z()));
  m_root->addChild(m_sliceZ->getTransform());

  redrawVoxelMesh();
  m_voxMesh->setUseDisplayList(false);

  setDisplayMode(m_displayMode);
}

osg::Vec3 CFDScene::getCenter()
{
  return osg::Vec3(m_voxSize->x() / 2, m_voxSize->y() / 2, m_voxSize->z() / 2);
}

CFDScene::CFDScene()
    : m_root(new osg::Group()),
      m_plot3d(0),
      m_plotGradient(0),
      m_displayQuantity(DisplayQuantity::TEMPERATURE),
      m_displayMode(DisplayMode::SLICE),
      m_voxMin(new osg::Vec3i(0, 0, 0)),
      m_voxMax(new osg::Vec3i(0, 0, 0)),
      m_voxSize(new osg::Vec3i(0, 0, 0)),
      m_plotMin(0),
      m_plotMax(0),
      m_slicePositions(new osg::Vec3i(0, 0, 0))

{
  setDisplayMode(DisplayMode::SLICE);

  // osg::ref_ptr<osg::Node> axes = osgDB::readRefNodeFile("assets/axes.osgt");
  // osg::ref_ptr<osg::Material> mat = new osg::Material();
  // mat->setAmbient(osg::Material::Face::FRONT_AND_BACK,
  //                 osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 10.0f);
  // mat->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
  //                 osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 10.0f);
  // mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
  //                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.2f);
  // mat->setColorMode(osg::Material::ColorMode::AMBIENT_AND_DIFFUSE);
  // osg::ref_ptr<osg::StateSet> stateset = axes->getOrCreateStateSet();
  // stateset->setAttribute(mat.get(), osg::StateAttribute::Values::ON);
  // stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
  // m_axesTransform = new osg::PositionAttitudeTransform;
  // m_axesTransform->setScale(osg::Vec3d(10, 10, 10));
  // m_axesTransform->addChild(axes);
}

void CFDScene::moveSlice(SliceRenderAxis::Enum axis, int inc)
{
  int pos;
  switch (axis)
  {
  case SliceRenderAxis::X_AXIS:
    switch (m_displayMode)
    {
    case DisplayMode::SLICE:
      pos = m_slicePositions->x();
      m_slicePositions->x() = (pos + inc <= m_voxSize->x() && pos + inc > 0) ? pos + inc : pos;
      m_sliceX->getTransform()->setPosition(osg::Vec3d((float)m_slicePositions->x(), 0, 0));
      break;
    case DisplayMode::VOX_GEOMETRY:
      pos = m_voxMax->x();
      m_voxMax->x() = (pos + inc <= (long)m_voxSize->x() && pos + inc >= 0) ? pos + inc : pos;
      break;
    }
    break;
  case SliceRenderAxis::Y_AXIS:
    switch (m_displayMode)
    {
    case DisplayMode::SLICE:
      pos = m_slicePositions->y();
      m_slicePositions->y() = (pos + inc <= m_voxSize->y() && pos + inc > 0) ? pos + inc : pos;
      m_sliceY->getTransform()->setPosition(osg::Vec3d(0, (float)m_slicePositions->y(), 0));
      break;
    case DisplayMode::VOX_GEOMETRY:
      pos = m_voxMax->y();
      m_voxMax->y() = (pos + inc <= (long)m_voxSize->y() && pos + inc >= 0) ? pos + inc : pos;
      break;
    }
    break;
  case SliceRenderAxis::Z_AXIS:
    switch (m_displayMode)
    {
    case DisplayMode::SLICE:
      pos = m_slicePositions->z();
      m_slicePositions->z() = (pos + inc <= m_voxSize->z() && pos + inc > 0) ? pos + inc : pos;
      m_sliceZ->getTransform()->setPosition(osg::Vec3d(0, 0, (float)m_slicePositions->z()));
      break;
    case DisplayMode::VOX_GEOMETRY:
      pos = m_voxMax->z();
      m_voxMax->z() = (pos + inc <= (long)m_voxSize->z() && pos + inc >= 0) ? pos + inc : pos;
      break;
    }
    break;
  }
  if (m_displayMode == DisplayMode::VOX_GEOMETRY)
    redrawVoxelMesh();
}
#include "CFDScene.hpp"

void CFDScene::redrawVoxelMesh()
{
  m_voxMesh->buildMesh(m_voxMin->x(), m_voxMax->x(),
                       m_voxMin->y(), m_voxMax->y(),
                       m_voxMin->z(), m_voxMax->z());
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
  cudaStreamCreateWithPriority(&renderStream, cudaStreamDefault, priority_low);

  m_voxMesh = mesh;
  m_voxSize = new osg::Vec3i(m_voxMesh->getSizeX(), m_voxMesh->getSizeY(), m_voxMesh->getSizeZ());
  m_voxMin = new osg::Vec3i(-1, -1, -1);
  m_voxMax = new osg::Vec3i(*m_voxSize);

  m_slicePositions = new osg::Vec3i(*m_voxSize);
  *m_slicePositions = *m_slicePositions / 2;

  m_sliceX = new SliceRender(SliceRenderAxis::X_AXIS, m_voxSize->y(), m_voxSize->z(), renderStream);
  m_sliceX->getTransform()->setAttitude(osg::Quat(osg::PI / 2, osg::Vec3d(0, 0, 1)));
  m_sliceX->getTransform()->setPosition(osg::Vec3d(m_slicePositions->x(), 0, 0));
  m_root->addChild(m_sliceX->getTransform());

  m_sliceY = new SliceRender(SliceRenderAxis::Y_AXIS, m_voxSize->x(), m_voxSize->z(), renderStream);
  m_sliceY->getTransform()->setAttitude(osg::Quat(0, osg::Vec3d(0, 0, 1)));
  m_sliceY->getTransform()->setPosition(osg::Vec3d(0, m_slicePositions->y(), 0));
  m_root->addChild(m_sliceY->getTransform());

  m_sliceZ = new SliceRender(SliceRenderAxis::Z_AXIS, m_voxSize->x(), m_voxSize->y(), renderStream);
  m_sliceZ->getTransform()->setAttitude(osg::Quat(-osg::PI / 2, osg::Vec3d(1, 0, 0)));
  m_sliceZ->getTransform()->setPosition(osg::Vec3d(0, 0, m_slicePositions->z()));
  m_root->addChild(m_sliceZ->getTransform());

  redrawVoxelMesh();
}

osg::Vec3 CFDScene::getCenter()
{
  return osg::Vec3(m_voxSize->x() / 2, m_voxSize->y() / 2, m_voxSize->z() / 2);
}

CFDScene::CFDScene()
{
  m_root = new osg::Group();
  osg::Geode *geode = new osg::Geode();
  m_voxGeo = new osg::Geometry();
  m_voxGeoTransform = new osg::PositionAttitudeTransform();

  geode->addDrawable(m_voxGeo);
  m_voxGeoTransform->addChild(geode);
  m_root->addChild(m_voxGeoTransform);

  m_voxGeoTransform->setPosition(osg::Vec3(0, 0, 0));
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
      pos = m_voxMin->x();
      m_voxMin->x() = (pos + inc <= (long)m_voxSize->x() && pos + inc >= 0) ? pos + inc : pos;
      redrawVoxelMesh();
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
      pos = m_voxMin->y();
      m_voxMin->y() = (pos + inc <= (long)m_voxSize->y() && pos + inc >= 0) ? pos + inc : pos;
      redrawVoxelMesh();
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
      pos = m_voxMin->z();
      m_voxMin->z() = (pos + inc <= (long)m_voxSize->z() && pos + inc >= 0) ? pos + inc : pos;
      redrawVoxelMesh();
      break;
    }
    break;
  }
}
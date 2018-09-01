#include "CFDScene.hpp"

void CFDScene::redrawVoxelMesh()
{
  m_voxMesh->buildMesh(*m_voxMin, *m_voxMax);
}

void CFDScene::setDisplayQuantity(DisplayQuantity::Enum quantity)
{
  m_displayQuantity = quantity;
  switch (quantity)
  {
  case DisplayQuantity::VELOCITY_NORM:
    m_plotMin = 0;
    m_plotMax = 0.2;
    break;
  case DisplayQuantity::DENSITY:
    m_plotMin = 1;
    m_plotMax = 1.1;
    break;
  case DisplayQuantity::TEMPERATURE:
    m_plotMin = 20;
    m_plotMax = 30;
    break;
  }
  // adjustColorGradient();
  if (m_sliceX)
    m_sliceX->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceY)
    m_sliceY->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceZ)
    m_sliceZ->setMinMax(m_plotMin, m_plotMax);
}

void CFDScene::setDisplayMode(DisplayMode::Enum mode)
{
  m_displayMode = mode;
  if (mode == DisplayMode::SLICE)
  {
    if (m_voxMesh)
    {
      m_voxMesh->setNodeMask(0);
      m_voxContour->setNodeMask(~0);
      m_voxFloor->setNodeMask(~0);
    }
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
    {
      m_voxMesh->setNodeMask(~0);
      m_voxContour->setNodeMask(0);
      m_voxFloor->setNodeMask(0);
    }
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

void CFDScene::setVoxelGeometry(std::shared_ptr<VoxelGeometry> voxels)
{
  // Clear the scene
  if (m_root->getNumChildren() > 0)
    m_root->removeChildren(0, m_root->getNumChildren());

  m_voxels = voxels;

  // Add voxel mesh to scene
  m_voxMesh = new VoxelMesh(voxels->data);
  m_voxSize = new osg::Vec3i(m_voxMesh->getSizeX(), m_voxMesh->getSizeY(), m_voxMesh->getSizeZ());
  m_voxMin = new osg::Vec3i(-1, -1, -1);
  m_voxMax = new osg::Vec3i(*m_voxSize + osg::Vec3i(-1, -1, -1));
  m_voxMesh->buildMesh(*m_voxMin, *m_voxMax);
  m_root->addChild(m_voxMesh->getTransform());

  m_voxContour = new VoxelContourMesh(voxels->data);
  m_voxContour->buildMesh();
  m_root->addChild(m_voxContour->getTransform());

  m_voxFloor = new VoxelFloorMesh(voxels->data);
  m_voxFloor->getTransform()->setAttitude(osg::Quat(-osg::PI / 2, osg::Vec3d(1, 0, 0)));
  m_voxFloor->getTransform()->setPosition(osg::Vec3d(0, 0, 0));
  m_root->addChild(m_voxFloor->getTransform());

  m_plot3d.erase(m_plot3d.begin(), m_plot3d.end());
  m_plot3d.reserve(m_voxMesh->getSize());
  m_plot3d.resize(m_voxMesh->getSize(), 0);

  // Add slice renderers to the scene
  m_slicePositions = new osg::Vec3i(*m_voxSize);
  *m_slicePositions = *m_slicePositions / 2;

  m_sliceX = new SliceRender(SliceRenderAxis::X_AXIS, m_voxSize->y(), m_voxSize->z(),
                             getPlot3d(), *m_voxSize);
  m_sliceX->setMinMax(m_plotMin, m_plotMax);
  m_sliceX->getTransform()->setAttitude(osg::Quat(osg::PI / 2, osg::Vec3d(0, 0, 1)));
  m_sliceX->getTransform()->setPosition(osg::Vec3d(m_slicePositions->x(), 0, 0));
  m_root->addChild(m_sliceX->getTransform());

  m_sliceY = new SliceRender(SliceRenderAxis::Y_AXIS, m_voxSize->x(), m_voxSize->z(),
                             getPlot3d(), *m_voxSize);
  m_sliceY->setMinMax(m_plotMin, m_plotMax);
  m_sliceY->getTransform()->setAttitude(osg::Quat(0, osg::Vec3d(0, 0, 1)));
  m_sliceY->getTransform()->setPosition(osg::Vec3d(0, m_slicePositions->y(), 0));
  m_root->addChild(m_sliceY->getTransform());

  m_sliceZ = new SliceRender(SliceRenderAxis::Z_AXIS, m_voxSize->x(), m_voxSize->y(),
                             getPlot3d(), *m_voxSize);
  m_sliceZ->setMinMax(m_plotMin, m_plotMax);
  m_sliceZ->getTransform()->setAttitude(osg::Quat(-osg::PI / 2, osg::Vec3d(1, 0, 0)));
  m_sliceZ->getTransform()->setPosition(osg::Vec3d(0, 0, m_slicePositions->z()));
  m_root->addChild(m_sliceZ->getTransform());

  setDisplayMode(m_displayMode);
}

bool CFDScene::pickVoxel(osg::Vec3d worldCoords)
{
  osg::Vec3i voxelCoords((int)worldCoords.x() + 1, (int)worldCoords.y() + 1, (int)worldCoords.z() + 1);
  voxel voxId = m_voxels->get(voxelCoords.x(), voxelCoords.y(), voxelCoords.z());

  if (voxId != VoxelType::EMPTY && voxId != VoxelType::FLUID && voxId > 0 && voxId < (int)m_voxels->voxdetail.size())
  {
    std::unordered_set<VoxelGeometryQuad> quads = m_voxels->quads.at(voxId);
    BoundaryCondition bc = m_voxels->voxdetail.at(voxId);
    std::cout << std::endl;
    for (const VoxelGeometryQuad &quad : quads)
      std::cout << quad.name << std::endl;
    std::cout << "pos: " << voxelCoords.x() << ", " << voxelCoords.y() << ", " << voxelCoords.z() << std::endl;
    std::cout << bc << std::endl;
    return true;
  }
  else
  {
    return false;
  }
}

osg::Vec3 CFDScene::getCenter()
{
  return osg::Vec3(m_voxSize->x() / 2, m_voxSize->y() / 2, m_voxSize->z() / 2);
}

CFDScene::CFDScene()
    : m_root(new osg::Group()),
      m_plot3d(0),
      m_plotGradient(0),
      m_voxMin(new osg::Vec3i(0, 0, 0)),
      m_voxMax(new osg::Vec3i(0, 0, 0)),
      m_voxSize(new osg::Vec3i(0, 0, 0)),
      m_plotMin(20),
      m_plotMax(30),
      m_slicePositions(new osg::Vec3i(0, 0, 0))

{
  setDisplayMode(DisplayMode::SLICE);
  setDisplayQuantity(DisplayQuantity::TEMPERATURE);
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
      m_slicePositions->x() = (pos + inc < m_voxSize->x() && pos + inc > 0) ? pos + inc : pos;
      m_sliceX->getTransform()->setPosition(osg::Vec3d((float)m_slicePositions->x(), 0, 0));
      break;
    case DisplayMode::VOX_GEOMETRY:
      pos = m_voxMax->x();
      m_voxMax->x() = (pos + inc < (long)m_voxSize->x() && pos + inc >= 0) ? pos + inc : pos;
      break;
    }
    break;
  case SliceRenderAxis::Y_AXIS:
    switch (m_displayMode)
    {
    case DisplayMode::SLICE:
      pos = m_slicePositions->y();
      m_slicePositions->y() = (pos + inc < m_voxSize->y() && pos + inc > 0) ? pos + inc : pos;
      m_sliceY->getTransform()->setPosition(osg::Vec3d(0, (float)m_slicePositions->y(), 0));
      break;
    case DisplayMode::VOX_GEOMETRY:
      pos = m_voxMax->y();
      m_voxMax->y() = (pos + inc < (long)m_voxSize->y() && pos + inc >= 0) ? pos + inc : pos;
      break;
    }
    break;
  case SliceRenderAxis::Z_AXIS:
    switch (m_displayMode)
    {
    case DisplayMode::SLICE:
      pos = m_slicePositions->z();
      m_slicePositions->z() = (pos + inc < m_voxSize->z() && pos + inc > 0) ? pos + inc : pos;
      m_sliceZ->getTransform()->setPosition(osg::Vec3d(0, 0, (float)m_slicePositions->z()));
      break;
    case DisplayMode::VOX_GEOMETRY:
      pos = m_voxMax->z();
      m_voxMax->z() = (pos + inc < (long)m_voxSize->z() && pos + inc >= 0) ? pos + inc : pos;
      break;
    }
    break;
  }
  if (m_displayMode == DisplayMode::VOX_GEOMETRY)
    redrawVoxelMesh();
}
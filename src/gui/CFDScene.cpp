#include "CFDScene.hpp"

void CFDScene::setDisplayQuantity(DisplayQuantity::Enum quantity) {
  m_displayQuantity = quantity;
  switch (quantity) {
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
      m_plotMax = 40;
      break;
  }
  if (m_sliceX) m_sliceX->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceY) m_sliceY->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceZ) m_sliceZ->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceGradient) m_sliceGradient->setMinMax(m_plotMin, m_plotMax);

  // Clear the histogram plot
  thrust::host_vector<real> h(HISTOGRAM_NUM_BINS);
  h[0] = 0.0;
  if (m_histogram) m_histogram->update(h);
}

void CFDScene::setColorScheme(ColorScheme::Enum colorScheme) {
  m_colorScheme = colorScheme;
  if (m_sliceX) m_sliceX->setColorScheme(m_colorScheme);
  if (m_sliceY) m_sliceY->setColorScheme(m_colorScheme);
  if (m_sliceZ) m_sliceZ->setColorScheme(m_colorScheme);
  if (m_sliceGradient) m_sliceGradient->setColorScheme(m_colorScheme);
}

void CFDScene::setAxesVisible(bool visible) {
  if (m_hud->getChildIndex(m_axes) == m_hud->getNumChildren()) {
    // Axes not in scene
    if (visible) m_hud->addChild(m_axes);
  } else {
    if (!visible) m_hud->removeChild(m_axes);
  }
}

void CFDScene::setDisplayMode(DisplayMode::Enum mode) {
  m_displayMode = mode;
  if (mode == DisplayMode::SLICE) {
    if (m_voxMesh) m_voxMesh->setNodeMask(0);
    if (m_voxContour) m_voxContour->setNodeMask(~0);
    if (m_voxFloor) m_voxFloor->setNodeMask(~0);
    if (m_marker) {
      m_marker->setNodeMask(0);
      m_marker->getLabel()->setNodeMask(0);
    }
    if (m_sliceX) m_sliceX->setNodeMask(~0);
    if (m_sliceY) m_sliceY->setNodeMask(~0);
    if (m_sliceZ) m_sliceZ->setNodeMask(~0);
    if (m_sliceGradient) m_sliceGradient->setNodeMask(~0);
    if (m_histogram) m_histogram->setNodeMask(~0);
    if (m_axes) m_axes->setNodeMask(0);
    if (m_partitionMesh) m_partitionMesh->setNodeMask(0);
    if (m_voxLabels) m_voxLabels->setNodeMask(m_showBCLabels ? ~0 : 0);
    if (m_avgLabels) m_avgLabels->setNodeMask(m_showAvgLabels ? ~0 : 0);
    if (m_avgs) m_avgs->setNodeMask(0);

  } else if (mode == DisplayMode::VOX_GEOMETRY) {
    if (m_voxMesh) m_voxMesh->setNodeMask(~0);
    if (m_voxContour) m_voxContour->setNodeMask(0);
    if (m_voxFloor) m_voxFloor->setNodeMask(0);
    if (m_marker) {
      m_marker->setNodeMask(0);
      m_marker->getLabel()->setNodeMask(0);
    }
    if (m_sliceX) m_sliceX->setNodeMask(0);
    if (m_sliceY) m_sliceY->setNodeMask(0);
    if (m_sliceZ) m_sliceZ->setNodeMask(0);
    if (m_sliceGradient) m_sliceGradient->setNodeMask(0);
    if (m_histogram) m_histogram->setNodeMask(0);
    if (m_axes) m_axes->setNodeMask(~0);
    if (m_partitionMesh) m_partitionMesh->setNodeMask(0);
    if (m_voxLabels) m_voxLabels->setNodeMask(m_showBCLabels ? ~0 : 0);
    if (m_avgLabels) m_avgLabels->setNodeMask(m_showAvgLabels ? ~0 : 0);
    if (m_avgs) m_avgs->setNodeMask(m_showAvgLabels ? ~0 : 0);

  } else if (mode == DisplayMode::DEVICES) {
    if (m_voxMesh) m_voxMesh->setNodeMask(0);
    if (m_voxContour) m_voxContour->setNodeMask(0);
    if (m_voxFloor) m_voxFloor->setNodeMask(0);
    if (m_marker) {
      m_marker->setNodeMask(0);
      m_marker->getLabel()->setNodeMask(0);
    }
    if (m_sliceX) m_sliceX->setNodeMask(0);
    if (m_sliceY) m_sliceY->setNodeMask(0);
    if (m_sliceZ) m_sliceZ->setNodeMask(0);
    if (m_sliceGradient) m_sliceGradient->setNodeMask(0);
    if (m_histogram) m_histogram->setNodeMask(0);
    if (m_axes) m_axes->setNodeMask(~0);
    if (m_partitionMesh) m_partitionMesh->setNodeMask(~0);
    if (m_voxLabels) m_voxLabels->setNodeMask(m_showBCLabels ? ~0 : 0);
    if (m_avgLabels) m_avgLabels->setNodeMask(0);
    if (m_avgs) m_avgs->setNodeMask(0);
  }
}

void CFDScene::adjustDisplayColors(real min,
                                   real max,
                                   const thrust::host_vector<real>& histogram) {
  m_plotMin = min;
  m_plotMax = max;
  // Adjust slice colors by min/max values
  if (m_sliceX) m_sliceX->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceY) m_sliceY->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceZ) m_sliceZ->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceGradient) m_sliceGradient->setMinMax(m_plotMin, m_plotMax);
  if (m_histogram) m_histogram->update(histogram);
}

void CFDScene::deleteVoxelGeometry() {
  if (getNumChildren() > 0) removeChildren(0, getNumChildren());
}

void CFDScene::setVoxelGeometry(std::shared_ptr<VoxelGeometry> voxels,
                                int numDevices) {
  std::cout << "Building graphics objects" << std::endl;

  // Clear the scene
  deleteVoxelGeometry();

  m_voxSize = new osg::Vec3i(voxels->getSizeX(), voxels->getSizeY(),
                             voxels->getSizeZ());
  m_voxMin = new osg::Vec3i(-1, -1, -1);
  m_voxMax = new osg::Vec3i(*m_voxSize - osg::Vec3i(1, 1, 1));

  m_voxLabels = new osg::Geode();
  for (std::pair<Eigen::Vector3i, std::string> element : voxels->getLabels()) {
    m_voxLabels->addChild(createBillboardText(element.first, element.second));
  }
  addChild(m_voxLabels);

  m_avgs = new osg::Geode();
  m_avgLabels = new osg::Geode();
  for (int i = 0; i < voxels->getSensors()->size(); i++) {
    VoxelVolume area = voxels->getSensors()->at(i);
    Eigen::Vector3i min = area.getMin();
    Eigen::Vector3i max = area.getMax();
    m_avgs->addChild(new VoxelAreaMesh(osg::Vec3i(min.x(), min.y(), min.z()),
                                       osg::Vec3i(max.x(), max.y(), max.z())));
    Eigen::Vector3i center((area.getMin() + area.getMax()) / 2);
    m_avgLabels->addChild(createBillboardText(center, area.getName()));
  }
  addChild(m_avgs);
  addChild(m_avgLabels);

  // Add voxel mesh to scene
  m_voxMesh = new VoxelMesh(voxels->getVoxelArray());
  addChild(m_voxMesh);

  // Add device partition mesh
  m_partitionMesh = new PartitionMesh(*m_voxMesh, numDevices, 0.3);
  addChild(m_partitionMesh);

  // Add voxel contour mesh
  m_voxContour = new VoxelContourMesh(*m_voxMesh);
  addChild(m_voxContour);

  // Add textured quad showing the floor
  m_voxFloor = new VoxelFloorMesh(voxels->getVoxelArray());
  m_voxFloor->getTransform()->setAttitude(
      osg::Quat(-osg::PI / 2, osg::Vec3d(1, 0, 0)));
  m_voxFloor->getTransform()->setPosition(osg::Vec3d(0, 0, 0));
  addChild(m_voxFloor->getTransform());

  // Add slice renderers to the scene
  m_slicePositions = new osg::Vec3i(*m_voxSize);
  *m_slicePositions = *m_slicePositions / 2;

  m_sliceX =
      new SliceRender(D3Q4::X_AXIS, m_voxSize->y(), m_voxSize->z(), *m_voxSize);
  m_sliceX->setMinMax(m_plotMin, m_plotMax);
  m_sliceX->getTransform()->setAttitude(
      osg::Quat(osg::PI / 2, osg::Vec3d(0, 0, 1)));
  m_sliceX->getTransform()->setPosition(
      osg::Vec3d(m_slicePositions->x(), 0, 0));
  m_sliceX->setColorScheme(m_colorScheme);
  addChild(m_sliceX->getTransform());

  m_sliceY =
      new SliceRender(D3Q4::Y_AXIS, m_voxSize->x(), m_voxSize->z(), *m_voxSize);
  m_sliceY->setMinMax(m_plotMin, m_plotMax);
  m_sliceY->getTransform()->setAttitude(osg::Quat(0, osg::Vec3d(0, 0, 1)));
  m_sliceY->getTransform()->setPosition(
      osg::Vec3d(0, m_slicePositions->y(), 0));
  m_sliceY->setColorScheme(m_colorScheme);
  addChild(m_sliceY->getTransform());

  m_sliceZ =
      new SliceRender(D3Q4::Z_AXIS, m_voxSize->x(), m_voxSize->y(), *m_voxSize);
  m_sliceZ->setMinMax(m_plotMin, m_plotMax);
  m_sliceZ->getTransform()->setAttitude(
      osg::Quat(-osg::PI / 2, osg::Vec3d(1, 0, 0)));
  m_sliceZ->getTransform()->setPosition(
      osg::Vec3d(0, 0, m_slicePositions->z()));
  m_sliceZ->setColorScheme(m_colorScheme);
  addChild(m_sliceZ->getTransform());

  setDisplayMode(m_displayMode);

  // // Voxel picking marker
  // addChild(m_marker->getTransform());

  std::cout << "Finished graphics objects" << std::endl;
}

bool CFDScene::selectVoxel(osg::Vec3d worldCoords) {
  // if (!voxels || m_displayMode != DisplayMode::VOX_GEOMETRY) return false;

  // m_marker->setNodeMask(~0);
  // m_marker->getLabel()->setNodeMask(~0);

  // osg::Vec3i voxelCoords(static_cast<int>(worldCoords.x()) + 1,
  //                        static_cast<int>(worldCoords.y()) + 1,
  //                        static_cast<int>(worldCoords.z()) + 1);
  // voxel voxId =
  //     voxels->get(voxelCoords.x(), voxelCoords.y(), voxelCoords.z());

  // if (voxId != VoxelType::EMPTY && voxId != VoxelType::FLUID && voxId > 0 &&
  //     voxId < voxels->getBoundaryConditions()->size()) {
  //   // Set the white marker voxel
  //   m_marker->getTransform()->setPosition(osg::Vec3d(voxelCoords.x() - 0.5f,
  //                                                    voxelCoords.y() - 0.5f,
  //                                                    voxelCoords.z() -
  //                                                    0.5f));
  //   // Show voxel info text label
  //   std::unordered_set<std::string> geometryNames =
  //       voxels->getObjectNamesById(voxId);
  //   BoundaryCondition bc = voxels->getBoundaryConditions()->at(voxId);
  //   std::stringstream ss;

  //   ss << "Pos: " << voxelCoords.x() << ", " << voxelCoords.y() << ", "
  //      << voxelCoords.z() << std::endl;
  //   ss << bc << std::endl;

  //   for (const std::string& name : geometryNames) {
  //     std::unordered_set<VoxelQuad> quads = voxels->getQuadsByName(name);
  //     int numQuads = quads.size();
  //     std::unordered_set<voxel> voxelsInObject =
  //         voxels->getVoxelsByName(name);
  //     int numVoxels = voxelsInObject.size();
  //     ss << std::endl
  //        << name << ": " << numQuads << " quads, " << numVoxels << " types";
  //   }
  //   m_marker->getLabel()->setText(ss.str());
  //   return true;
  // }
  return false;
}

void CFDScene::resize(int width, int height) {
  m_hud->resize(width, height);
  if (m_sliceGradient) m_sliceGradient->resize(width);
  if (m_histogram) m_histogram->resize(width);
  if (m_axes) m_axes->resize(width, height);
}

osg::Vec3 CFDScene::getCenter() {
  return osg::Vec3(m_voxSize->x() / 2, m_voxSize->y() / 2, m_voxSize->z() / 2);
}

CFDScene::CFDScene()
    : osg::Geode(),
      m_voxMin(new osg::Vec3i(0, 0, 0)),
      m_voxMax(new osg::Vec3i(0, 0, 0)),
      m_voxSize(new osg::Vec3i(0, 0, 0)),
      m_showBCLabels(false),
      m_showAvgLabels(false),
      m_plotMin(0),
      m_plotMax(0),
      m_slicePositions(new osg::Vec3i(0, 0, 0)),
      m_hud(new CFDHud(1, 1)),
      m_histogram(new HistogramMesh()),
      m_marker(new VoxelMarker()),
      m_colorScheme(ColorScheme::PARAVIEW) {
  m_sliceGradient = new SliceRenderGradient();
  m_sliceGradient->setMinMax(m_plotMin, m_plotMax);
  m_sliceGradient->setColorScheme(m_colorScheme);

  osg::StateSet* stateset = getOrCreateStateSet();
  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);

  m_hud->addChild(m_histogram->getTransform());
  m_hud->addChild(m_sliceGradient->getTransform());
  for (int i = 0; i < m_sliceGradient->getNumLabels(); i++)
    m_hud->addDrawable(m_sliceGradient->getLabel(i));
  m_hud->addDrawable(m_marker->getLabel());

  m_axes = new AxesMesh();
  m_hud->addChild(m_axes);

  setDisplayMode(DisplayMode::SLICE);
  setDisplayQuantity(DisplayQuantity::TEMPERATURE);
}

void CFDScene::moveSlice(D3Q4::Enum axis, int inc) {
  if (inc == 0) return;
  int pos;
  switch (axis) {
    case D3Q4::X_AXIS:
      switch (m_displayMode) {
        case DisplayMode::SLICE:
          pos = m_slicePositions->x();
          m_slicePositions->x() =
              (pos + inc < m_voxSize->x() && pos + inc > 0) ? pos + inc : pos;
          m_sliceX->getTransform()->setPosition(
              osg::Vec3d(static_cast<float>(m_slicePositions->x()), 0, 0));
          break;
        case DisplayMode::DEVICES:
          // [[fallthrough]];
        case DisplayMode::VOX_GEOMETRY:
          pos = m_voxMin->x();
          m_voxMin->x() =
              (pos + inc < m_voxSize->x() && pos + inc >= 0) ? pos + inc : pos;
          break;
      }
      break;
    case D3Q4::Y_AXIS:
      switch (m_displayMode) {
        case DisplayMode::SLICE:
          pos = m_slicePositions->y();
          m_slicePositions->y() =
              (pos + inc < m_voxSize->y() && pos + inc > 0) ? pos + inc : pos;
          m_sliceY->getTransform()->setPosition(
              osg::Vec3d(0, static_cast<float>(m_slicePositions->y()), 0));
          break;
        case DisplayMode::DEVICES:
          // [[fallthrough]];
        case DisplayMode::VOX_GEOMETRY:
          pos = m_voxMin->y();
          m_voxMin->y() =
              (pos + inc < m_voxSize->y() && pos + inc >= 0) ? pos + inc : pos;
          break;
      }
      break;
    case D3Q4::Z_AXIS:
      switch (m_displayMode) {
        case DisplayMode::SLICE:
          pos = m_slicePositions->z();
          m_slicePositions->z() =
              (pos + inc < m_voxSize->z() && pos + inc > 0) ? pos + inc : pos;
          m_sliceZ->getTransform()->setPosition(
              osg::Vec3d(0, 0, static_cast<float>(m_slicePositions->z())));
          break;
        case DisplayMode::DEVICES:
          // [[fallthrough]];
        case DisplayMode::VOX_GEOMETRY:
          pos = m_voxMax->z();
          m_voxMax->z() =
              (pos + inc < m_voxSize->z() && pos + inc >= 0) ? pos + inc : pos;
          break;
      }
      break;
  }
  if (m_displayMode == DisplayMode::VOX_GEOMETRY ||
      m_displayMode == DisplayMode::DEVICES)
    m_voxMesh->crop(*m_voxMin, *m_voxMax);
}

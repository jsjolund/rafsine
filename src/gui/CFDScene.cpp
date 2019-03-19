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
      m_plotMax = 30;
      break;
  }
  if (m_sliceX) m_sliceX->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceY) m_sliceY->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceZ) m_sliceZ->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceGradient) m_sliceGradient->setMinMax(m_plotMin, m_plotMax);
}

void CFDScene::setColorScheme(ColorScheme::Enum colorScheme) {
  if (m_sliceX) m_sliceX->setColorScheme(colorScheme);
  if (m_sliceY) m_sliceY->setColorScheme(colorScheme);
  if (m_sliceZ) m_sliceZ->setColorScheme(colorScheme);
  if (m_sliceGradient) m_sliceGradient->setColorScheme(colorScheme);
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
    if (m_axes) m_axes->setNodeMask(0);
    if (m_subLatticeMesh) m_subLatticeMesh->setNodeMask(0);
    if (m_labels) m_labels->setNodeMask(m_showLabels ? ~0 : 0);
    if (m_sensors) m_sensors->setNodeMask(0);

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
    if (m_axes) m_axes->setNodeMask(~0);
    if (m_subLatticeMesh) m_subLatticeMesh->setNodeMask(0);
    if (m_labels) m_labels->setNodeMask(m_showLabels ? ~0 : 0);
    if (m_sensors) m_sensors->setNodeMask(m_showSensors ? ~0 : 0);

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
    if (m_axes) m_axes->setNodeMask(~0);
    if (m_subLatticeMesh) m_subLatticeMesh->setNodeMask(~0);
    if (m_labels) m_labels->setNodeMask(m_showLabels ? ~0 : 0);
    if (m_sensors) m_sensors->setNodeMask(0);
  }
}

void CFDScene::adjustDisplayColors(real min, real max) {
  m_plotMin = min;
  m_plotMax = max;
  if (m_plot3d.size() == 0) return;
  // Adjust slice colors by min/max values
  if (m_sliceX) m_sliceX->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceY) m_sliceY->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceZ) m_sliceZ->setMinMax(m_plotMin, m_plotMax);
  if (m_sliceGradient) m_sliceGradient->setMinMax(m_plotMin, m_plotMax);
}

void CFDScene::setVoxelGeometry(std::shared_ptr<VoxelGeometry> voxels,
                                int numDevices) {
  std::cout << "Building graphics objects" << std::endl;

  // Clear the scene
  if (getNumChildren() > 0) removeChildren(0, getNumChildren());

  m_voxSize = new osg::Vec3i(voxels->getNx(), voxels->getNy(), voxels->getNz());
  m_voxMin = new osg::Vec3i(-1, -1, -1);
  m_voxMax = new osg::Vec3i(*m_voxSize - osg::Vec3i(1, 1, 1));

  m_labels = new osg::Geode();
  for (std::pair<glm::ivec3, std::string> element : voxels->getLabels()) {
    m_labels->addChild(createBillboardText(element.first, element.second));
  }
  addChild(m_labels);

  m_sensors = new osg::Geode();
  for (int i = 0; i < voxels->getSensors()->size(); i++) {
    VoxelArea area = voxels->getSensors()->at(i);
    m_sensors->addChild(new VoxelAreaMesh(area.getMin(), area.getMax()));
  }
  addChild(m_sensors);

  // // Add voxel mesh to scene
  // m_voxMesh = new VoxelMesh(voxels->getVoxelArray());
  // addChild(m_voxMesh->getTransform());

  // // Add device subLattice mesh
  // m_subLatticeMesh = new SubLatticeMesh(*m_voxMesh, numDevices, 0.3);
  // addChild(m_subLatticeMesh);

  // // Add voxel contour mesh
  // m_voxContour = new VoxelContourMesh(*m_voxMesh);
  // addChild(m_voxContour->getTransform());

  // Add textured quad showing the floor
  m_voxFloor = new VoxelFloorMesh(voxels->getVoxelArray());
  m_voxFloor->getTransform()->setAttitude(
      osg::Quat(-osg::PI / 2, osg::Vec3d(1, 0, 0)));
  m_voxFloor->getTransform()->setPosition(osg::Vec3d(0, 0, 0));
  addChild(m_voxFloor->getTransform());

  // Resize the plot
  m_plot3d.erase(m_plot3d.begin(), m_plot3d.end());
  m_plot3d.reserve(voxels->getSize());
  m_plot3d.resize(voxels->getSize(), 0);

  // Add slice renderers to the scene
  m_slicePositions = new osg::Vec3i(*m_voxSize);
  *m_slicePositions = *m_slicePositions / 2;

  m_sliceX = new SliceRender(D3Q7::X_AXIS_POS, m_voxSize->y(), m_voxSize->z(),
                             gpu_ptr(), *m_voxSize);
  m_sliceX->setMinMax(m_plotMin, m_plotMax);
  m_sliceX->getTransform()->setAttitude(
      osg::Quat(osg::PI / 2, osg::Vec3d(0, 0, 1)));
  m_sliceX->getTransform()->setPosition(
      osg::Vec3d(m_slicePositions->x(), 0, 0));
  addChild(m_sliceX->getTransform());

  m_sliceY = new SliceRender(D3Q7::Y_AXIS_POS, m_voxSize->x(), m_voxSize->z(),
                             gpu_ptr(), *m_voxSize);
  m_sliceY->setMinMax(m_plotMin, m_plotMax);
  m_sliceY->getTransform()->setAttitude(osg::Quat(0, osg::Vec3d(0, 0, 1)));
  m_sliceY->getTransform()->setPosition(
      osg::Vec3d(0, m_slicePositions->y(), 0));
  addChild(m_sliceY->getTransform());

  m_sliceZ = new SliceRender(D3Q7::Z_AXIS_POS, m_voxSize->x(), m_voxSize->y(),
                             gpu_ptr(), *m_voxSize);
  m_sliceZ->setMinMax(m_plotMin, m_plotMax);
  m_sliceZ->getTransform()->setAttitude(
      osg::Quat(-osg::PI / 2, osg::Vec3d(1, 0, 0)));
  m_sliceZ->getTransform()->setPosition(
      osg::Vec3d(0, 0, m_slicePositions->z()));
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
  if (m_axes) m_axes->resize(width, height);
}

osg::Vec3 CFDScene::getCenter() {
  return osg::Vec3(m_voxSize->x() / 2, m_voxSize->y() / 2, m_voxSize->z() / 2);
}

CFDScene::CFDScene()
    : osg::Geode(),
      m_plot3d(0),
      m_plotGradient(0),
      m_voxMin(new osg::Vec3i(0, 0, 0)),
      m_voxMax(new osg::Vec3i(0, 0, 0)),
      m_voxSize(new osg::Vec3i(0, 0, 0)),
      m_showLabels(true),
      m_showSensors(true),
      m_plotMin(20),
      m_plotMax(30),
      m_slicePositions(new osg::Vec3i(0, 0, 0)),
      m_hud(new CFDHud(1, 1)),
      m_marker(new VoxelMarker()) {
  m_sliceGradient = new SliceRenderGradient();
  m_sliceGradient->setMinMax(m_plotMin, m_plotMax);

  osg::StateSet* stateSet = getOrCreateStateSet();
  stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);

  m_hud->addChild(m_sliceGradient->getTransform());
  for (int i = 0; i < m_sliceGradient->getNumLabels(); i++)
    m_hud->addDrawable(m_sliceGradient->getLabel(i));
  m_hud->addDrawable(m_marker->getLabel());

  m_axes = new AxesMesh();
  m_hud->addChild(m_axes);

  setDisplayMode(DisplayMode::SLICE);
  setDisplayQuantity(DisplayQuantity::TEMPERATURE);
}

void CFDScene::moveSlice(D3Q7::Enum axis, int inc) {
  if (inc == 0) return;
  int pos;
  switch (axis) {
    case D3Q7::X_AXIS_POS:
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
    case D3Q7::Y_AXIS_POS:
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
    case D3Q7::Z_AXIS_POS:
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

#include "PartitionMesh.hpp"

void PartitionMesh::setProperties(osg::ref_ptr<osg::ShapeDrawable> drawable) {
  osg::ref_ptr<osg::StateSet> stateset = drawable->getOrCreateStateSet();
  // Filled ploygons
  osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
  polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);
  stateset->setAttributeAndModes(
      polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
  // Lighting
  stateset->setMode(GL_LIGHTING,
                    osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
  // Material
  osg::ref_ptr<osg::Material> material = new osg::Material;
  material->setAmbient(osg::Material::Face::FRONT_AND_BACK,
                       osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 1.0f);
  material->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
                       osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.5f);
  material->setEmission(osg::Material::Face::FRONT_AND_BACK,
                        osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.1f);
  material->setColorMode(osg::Material::ColorMode::AMBIENT_AND_DIFFUSE);
  stateset->setAttributeAndModes(
      material, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
  // Transparency
  stateset->setAttributeAndModes(
      new osg::BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
}

void PartitionMesh::addLabel(osg::Vec3d center, std::string content) {
  osg::ref_ptr<osg::PositionAttitudeTransform> transform =
      new osg::PositionAttitudeTransform();
  osg::ref_ptr<osgText::Text> text = new BillboardText();
  text->setBoundingBoxColor(osg::Vec4(0.0f, 0.0f, 0.0f, 0.5f));
  text->setCharacterSizeMode(osgText::Text::SCREEN_COORDS);
  text->setDrawMode(osgText::Text::TEXT |
                    // osgText::Text::ALIGNMENT |
                    osgText::Text::FILLEDBOUNDINGBOX);
  text->setAlignment(osgText::Text::LEFT_TOP);
  transform->addChild(text);
  transform->setPosition(center);
  addChild(transform);
  text->setText(content);
}

PartitionMesh::PartitionMesh(const VoxelMesh& voxMesh,
                             int numDevices,
                             float alpha)
    : osg::Geode(), m_voxMesh(new VoxelMesh(voxMesh)) {
  DistributedLattice lattice(m_voxMesh->getSizeX(), m_voxMesh->getSizeY(),
                             m_voxMesh->getSizeZ(), numDevices);
  const int numPartitions = lattice.getNumPartitionsTotal();
  for (int i = 0; i < numPartitions; i++) {
    Partition partition = lattice.getPartitions().at(i);

    Eigen::Vector3i min = partition.getMin();
    Eigen::Vector3i size = partition.getExtents();
    Eigen::Vector3f c =
        Eigen::Vector3f(min.x(), min.y(), min.z()) +
        Eigen::Vector3f(size.x() * 0.5f, size.y() * 0.5f, size.z() * 0.5f);

    // Create boxes
    {
      osg::ref_ptr<osg::ShapeDrawable> drawable =
          new osg::ShapeDrawable(new osg::Box(osg::Vec3d(c.x(), c.y(), c.z()),
                                              size.x(), size.y(), size.z()));
      osg::Vec4 color = m_colorSet.getColor(i + 2);
      color.a() *= alpha;
      drawable->setColor(color);
      addDrawable(drawable);
      // Set box visibility properties
      setProperties(drawable);
    }

    // Create labels
    std::stringstream ss;
    ss << "GPU" << lattice.getPartitionDevice(partition);
    addLabel(osg::Vec3d(c.x(), c.y(), c.z()), ss.str());
  }
  osg::Vec3i voxMin(2, 2, 2);
  osg::Vec3i voxMax(m_voxMesh->getSizeX() - 3, m_voxMesh->getSizeY() - 3,
                    m_voxMesh->getSizeZ() - 3);
  m_voxMesh->crop(voxMin, voxMax);
  addChild(m_voxMesh);
}

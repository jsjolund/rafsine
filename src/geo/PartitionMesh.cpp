#include "PartitionMesh.hpp"

PartitionMesh::~PartitionMesh() { delete m_colorSet; }

void PartitionMesh::setTransparent(osg::ref_ptr<osg::ShapeDrawable> drawable) {
  osg::ref_ptr<osg::StateSet> stateset = drawable->getOrCreateStateSet();
  osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
  polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);
  stateset->setAttributeAndModes(
      polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
  stateset->setMode(GL_LIGHTING,
                    osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
  osg::ref_ptr<osg::Material> material = new osg::Material;
  stateset->setAttributeAndModes(
      material, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
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
  text->setDrawMode(osgText::Text::TEXT | osgText::Text::ALIGNMENT |
                    osgText::Text::FILLEDBOUNDINGBOX);
  text->setAlignment(osgText::Text::LEFT_TOP);
  transform->addChild(text);
  transform->setPosition(center);
  addChild(transform);
  text->setText(content);
}

PartitionMesh::PartitionMesh(unsigned int Q, unsigned int latticeSizeX,
                             unsigned int latticeSizeY,
                             unsigned int latticeSizeZ, unsigned int partitions)
    : Topology(Q, latticeSizeX, latticeSizeY, latticeSizeZ, partitions),
      osg::Geode(),
      m_colorSet(new ColorSet()) {
  const int numPartitions = getNumPartitionsTotal();
  for (int i = 0; i < numPartitions; i++) {
    Partition partition = m_partitions[i];

    glm::ivec3 min = partition.getLatticeMin();
    glm::ivec3 size = partition.getLatticeDims();
    glm::vec3 c =
        glm::vec3(min) + glm::vec3(size.x * 0.5f, size.y * 0.5f, size.z * 0.5f);

    // Create boxes
    {
      osg::ref_ptr<osg::ShapeDrawable> drawable = new osg::ShapeDrawable(
          new osg::Box(osg::Vec3d(c.x, c.y, c.z), size.x, size.y, size.z));
      osg::Vec4 color = m_colorSet->getColor(i + 2);
      //   color.a() *= 0.2;
      drawable->setColor(color);
      addDrawable(drawable);
      // Set box visibility properties
      //   setTransparent(drawable);
    }

    // Create labels
    std::stringstream ss;
    ss << "GPU" << i;  // TODO(Don't assign index like this...)
    addLabel(osg::Vec3d(c.x, c.y, c.z), ss.str());

    // for (int q = 1; q < getQ(); q++) {
    //   Partition neighbour = getNeighbour(partition, q);
    //   std::vector<PartitionSegment> segments =
    //   m_segments[partition][neighbour]; for (PartitionSegment segment :
    //   segments) {
    //     glm::vec3 origin =
    //         glm::vec3(min) +
    //         glm::vec3(segment.m_src.x, segment.m_src.y, segment.m_src.z) +
    //         glm::vec3(0.5, 0.5, 0.5);
    //     std::cout << origin << std::endl;
    //     osg::ref_ptr<osg::ShapeDrawable> drawable = new osg::ShapeDrawable(
    //         new osg::Box(osg::Vec3d(origin.x, origin.y, origin.z), 1, 1, 1));
    //     drawable->setColor(m_colorSet->getColor(q));
    //     addDrawable(drawable);
    //   }
    // }
  }
}

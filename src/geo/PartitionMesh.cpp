#include "PartitionMesh.hpp"

PartitionMesh::~PartitionMesh() { delete m_colorSet; }

PartitionMesh::PartitionMesh(unsigned int latticeSizeX,
                             unsigned int latticeSizeY,
                             unsigned int latticeSizeZ, unsigned int partitions)
    : Topology(1, latticeSizeX, latticeSizeY, latticeSizeZ, partitions),
      osg::Geode(),
      m_colorSet(new ColorSet()) {
  std::unordered_map<Partition, osg::Vec4> colorMap;

  const int numPartitions = getNumPartitionsTotal();
  for (int i = 0; i < numPartitions; i++) {
    Partition partition = m_partitions[i];

    // Create boxes
    float cx = partition.getLatticeMin().x + partition.getLatticeDims().x * 0.5;
    float cy = partition.getLatticeMin().y + partition.getLatticeDims().y * 0.5;
    float cz = partition.getLatticeMin().z + partition.getLatticeDims().z * 0.5;
    osg::Vec3d center(cx, cy, cz);
    osg::ref_ptr<osg::ShapeDrawable> drawable =
        new osg::ShapeDrawable(new osg::Box(
            center, partition.getLatticeDims().x, partition.getLatticeDims().y,
            partition.getLatticeDims().z));

    osg::Vec4 color = m_colorSet->getColor(i + 2);
    drawable->setColor(
        osg::Vec4f(color.r(), color.g(), color.b(), color.a() * 0.2));
    colorMap[partition] = color;

    addDrawable(drawable);

    // Set box visibility properties
    osg::ref_ptr<osg::StateSet> stateset = drawable->getOrCreateStateSet();
    osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
    polymode->setMode(osg::PolygonMode::FRONT, osg::PolygonMode::FILL);
    stateset->setAttributeAndModes(
        polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    stateset->setMode(GL_LIGHTING,
                      osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
    osg::ref_ptr<osg::Material> material = new osg::Material;
    stateset->setAttributeAndModes(
        material, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    stateset->setAttributeAndModes(
        new osg::BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    stateset->setMode(GL_BLEND, osg::StateAttribute::ON);

    // Create labels
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

    // TODO(GPU is not necessarily set like this...)
    addChild(transform);
    std::stringstream ss;
    ss << "GPU" << i;
    text->setText(ss.str());
  }
}

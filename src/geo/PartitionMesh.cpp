#include "PartitionMesh.hpp"

PartitionMesh::~PartitionMesh() { delete m_colorSet; }

PartitionMesh::PartitionMesh(unsigned int latticeSizeX,
                             unsigned int latticeSizeY,
                             unsigned int latticeSizeZ,
                             unsigned int subdivisions)
    : Topology(latticeSizeX, latticeSizeY, latticeSizeX, subdivisions),
      osg::Geode(),
      m_transform(new osg::PositionAttitudeTransform()),
      m_colorSet(new ColorSet()) {
  std::unordered_map<Partition, osg::Vec4> colorMap;

  for (int i = 0; i < getNumPartitions(); i++) {
    Partition *partition = m_partitions[i];

    float cx = partition->getMin().x + partition->getN().x * 0.5;
    float cy = partition->getMin().y + partition->getN().y * 0.5;
    float cz = partition->getMin().z + partition->getN().z * 0.5;
    osg::Vec3d center(cx, cy, cz);
    osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(new osg::Box(
        center, partition->getNx(), partition->getNy(), partition->getNz()));

    osg::Vec4 color = m_colorSet->getColor(i + 2);
    sd->setColor(osg::Vec4f(color.r(), color.g(), color.b(), color.a() * 0.5));
    colorMap[*partition] = color;

    addDrawable(sd);

    // Show partition as lines, no lighting effect
    osg::ref_ptr<osg::StateSet> stateset = sd->getOrCreateStateSet();
    osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
    polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);
    stateset->setAttributeAndModes(
        polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    osg::ref_ptr<osg::Material> material = new osg::Material;
    stateset->setMode(GL_LIGHTING,
                      osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
    stateset->setAttributeAndModes(
        material, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    stateset->setAttributeAndModes(
        new osg::BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
  }

  for (int i = 0; i < getNumPartitions(); i++) {
    Partition *partition = m_partitions[i];
    for (std::pair<glm::ivec3, Partition *> keyValue :
         partition->m_neighbours) {
      glm::ivec3 direction = keyValue.first;
      Partition *neighbour = keyValue.second;

      std::vector<glm::ivec3> srcPoints;
      std::vector<glm::ivec3> haloPoints;
      partition->getHalo(direction, &srcPoints, &haloPoints);
      for (glm::ivec3 haloPoint : haloPoints) {
        osg::Vec3d center(haloPoint.x + 0.5, haloPoint.y + 0.5,
                          haloPoint.z + 0.5);
        osg::ref_ptr<osg::ShapeDrawable> sd =
            new osg::ShapeDrawable(new osg::Box(center, 1, 1, 1));
        osg::Vec4 color = colorMap[*neighbour];
        sd->setColor(color);

        addDrawable(sd);

        // Show halo as points, no lighting effect
        osg::ref_ptr<osg::StateSet> stateset = sd->getOrCreateStateSet();
        osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
        polymode->setMode(osg::PolygonMode::FRONT_AND_BACK,
                          osg::PolygonMode::POINT);
        stateset->setAttributeAndModes(
            polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
        stateset->setAttribute(new osg::Point(5.0f), osg::StateAttribute::ON);
        osg::ref_ptr<osg::Material> material = new osg::Material;
        stateset->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE |
                                           osg::StateAttribute::OFF);
        stateset->setAttributeAndModes(
            material, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
      }
    }
  }
}

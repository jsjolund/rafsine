#include "PartitionTopology.hpp"

static void recursiveSubpartition(int divisions, glm::ivec3 *partitionCount, std::vector<Partition *> *partitions)
{
  if (divisions > 0)
  {
    std::vector<Partition *> oldPartitions;
    oldPartitions.insert(oldPartitions.end(), partitions->begin(), partitions->end());
    partitions->clear();
    const Partition::Enum axis = oldPartitions.at(0)->getDivisionAxis();
    if (axis == Partition::X_AXIS)
      partitionCount->x *= 2;
    if (axis == Partition::Y_AXIS)
      partitionCount->y *= 2;
    if (axis == Partition::Z_AXIS)
      partitionCount->z *= 2;

    for (Partition *p : oldPartitions)
    {
      glm::ivec3 a_min = p->getMin(), a_max = p->getMax(), b_min = p->getMin(), b_max = p->getMax();
      int nx = p->getNx(), ny = p->getNy(), nz = p->getNz();
      switch (axis)
      {
      case Partition::X_AXIS:
        a_max.x = p->getMin().x + std::ceil(1.0 * nx / 2);
        b_min.x = a_max.x;
        break;
      case Partition::Y_AXIS:
        a_max.y = p->getMin().y + std::ceil(1.0 * ny / 2);
        b_min.y = a_max.y;
        break;
      case Partition::Z_AXIS:
        a_max.z = p->getMin().z + std::ceil(1.0 * nz / 2);
        b_min.z = a_max.z;
        break;
      default:
        break;
      }
      partitions->push_back(new Partition(a_min, a_max));
      partitions->push_back(new Partition(b_min, b_max));
    }
    recursiveSubpartition(divisions - 1, partitionCount, partitions);
  }
}

Partition::Enum Partition::getDivisionAxis()
{
  int nx = getNx(), ny = getNy(), nz = getNz();
  int xz = nx * nz, yz = ny * nz, xy = nx * ny;
  if (xy <= xz && xy <= yz)
    return Partition::Z_AXIS;
  if (xz <= yz && xz <= xy)
    return Partition::Y_AXIS;
  else
    return Partition::X_AXIS;
}

void Topology::buildMesh()
{
  for (int i = 0; i < size(); i++)
  {
    Partition *p = m_partitions[i];

    float cx = p->getMin().x + p->getNx() * 0.5f;
    float cy = p->getMin().y + p->getNy() * 0.5f;
    float cz = p->getMin().z + p->getNz() * 0.5f;
    osg::Vec3d center(cx, cy, cz);
    osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(new osg::Box(center, p->getNx(), p->getNy(), p->getNz()));

    glm::vec3 color = m_colorSet->getColor(i + 2);
    sd->setColor(osg::Vec4f(color.r, color.g, color.b, 1.0f));

     osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(sd);
    m_root->addChild(geode);
  }
}

Topology::Topology(int latticeSizeX, int latticeSizeY, int latticeSizeZ, int subdivisions)
    : m_colorSet(new ColorSet()),
      m_root(new osg::Group()),
      m_partitionCount(glm::ivec3(1, 1, 1)),
      m_latticeSize(glm::ivec3(latticeSizeX, latticeSizeY, latticeSizeZ))
{
  Partition *p = new Partition(glm::ivec3(0, 0, 0), m_latticeSize);
  m_partitions.push_back(p);
  if (subdivisions > 0)
    recursiveSubpartition(subdivisions, &m_partitionCount, &m_partitions);

  std::sort(m_partitions.begin(), m_partitions.end(),
            [](Partition *a, Partition *b) {
              if (a->getMin().z != b->getMin().z)
                return a->getMin().z < b->getMin().z;
              if (a->getMin().y != b->getMin().y)
                return a->getMin().y < b->getMin().y;
              return a->getMin().x < b->getMin().x;
            });

  int totalVol = 0;
  for (int x = 0; x < getNx(); x++)
    for (int y = 0; y < getNy(); y++)
      for (int z = 0; z < getNz(); z++)
      {
        Partition *p = operator()(x, y, z);
        totalVol += p->getVolume();
      }

  assert(totalVol == m_latticeSize.x * m_latticeSize.y * m_latticeSize.z);
  assert(1 << subdivisions == m_partitionCount.x * m_partitionCount.y * m_partitionCount.z);
  assert(1 << subdivisions == size());

  buildMesh();
}
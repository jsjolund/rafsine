#include "PartitionTopology.hpp"

bool operator==(Partition const &a, Partition const &b)
{
  return (a.getMin().x == b.getMin().x && a.getMin().y == b.getMin().y && a.getMin().z == b.getMin().z && a.getMax().x == b.getMax().x && a.getMax().y == b.getMax().y && a.getMax().z == b.getMax().z);
}

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

    for (Partition *partition : oldPartitions)
    {
      glm::ivec3 a_min = partition->getMin(), a_max = partition->getMax(), b_min = partition->getMin(), b_max = partition->getMax();
      int nx = partition->getNx(), ny = partition->getNy(), nz = partition->getNz();
      switch (axis)
      {
      case Partition::X_AXIS:
        a_max.x = partition->getMin().x + std::ceil(1.0 * nx / 2);
        b_min.x = a_max.x;
        break;
      case Partition::Y_AXIS:
        a_max.y = partition->getMin().y + std::ceil(1.0 * ny / 2);
        b_min.y = a_max.y;
        break;
      case Partition::Z_AXIS:
        a_max.z = partition->getMin().z + std::ceil(1.0 * nz / 2);
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

void Partition::getHalo(glm::ivec3 direction, std::vector<glm::ivec3> *haloPoints)
{
  glm::ivec3 origin, dir1, dir2;
  // 6 faces
  if (direction == glm::ivec3(1, 0, 0))
  {
    origin = glm::ivec3(m_max.x, m_min.y, m_min.z);
    dir1 = glm::ivec3(0, m_max.y - m_min.y, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  }
  else if (direction == glm::ivec3(-1, 0, 0))
  {
    origin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(-1, 0, 0);
    dir1 = glm::ivec3(0, m_max.y - m_min.y, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  }
  else if (direction == glm::ivec3(0, 1, 0))
  {
    origin = glm::ivec3(m_min.x, m_max.y, m_min.z);
    dir1 = glm::ivec3(m_max.x - m_min.x, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  }
  else if (direction == glm::ivec3(0, -1, 0))
  {
    origin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(0, -1, 0);
    dir1 = glm::ivec3(m_max.x - m_min.x, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  }
  else if (direction == glm::ivec3(0, 0, 1))
  {
    origin = glm::ivec3(m_min.x, m_min.y, m_max.z);
    dir1 = glm::ivec3(m_max.x - m_min.x, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  }
  else if (direction == glm::ivec3(0, 0, -1))
  {
    origin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(0, 0, -1);
    dir1 = glm::ivec3(m_max.x - m_min.x, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  }
  // 12 edges
  else if (direction == glm::ivec3(1, 1, 0))
  {
    origin = glm::ivec3(m_max.x, m_max.y, m_min.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  }
  else if (direction == glm::ivec3(-1, -1, 0))
  {
    origin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(-1, -1, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  }
  else if (direction == glm::ivec3(1, -1, 0))
  {
    origin = glm::ivec3(m_max.x, m_min.y, m_min.z) + glm::ivec3(0, -1, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  }
  else if (direction == glm::ivec3(-1, 1, 0))
  {
    origin = glm::ivec3(m_min.x, m_max.y, m_min.z) + glm::ivec3(-1, 0, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 0, m_max.z - m_min.z);
  }
  else if (direction == glm::ivec3(1, 0, 1))
  {
    origin = glm::ivec3(m_max.x, m_min.y, m_max.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  }
  else if (direction == glm::ivec3(-1, 0, -1))
  {
    origin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(-1, 0, -1);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  }
  else if (direction == glm::ivec3(1, 0, -1))
  {
    origin = glm::ivec3(m_max.x, m_min.y, m_min.z) + glm::ivec3(0, 0, -1);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  }
  else if (direction == glm::ivec3(-1, 0, 1))
  {
    origin = glm::ivec3(m_min.x, m_min.y, m_max.z) + glm::ivec3(-1, 0, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, m_max.y - m_min.y, 0);
  }
  else if (direction == glm::ivec3(0, 1, 1))
  {
    origin = glm::ivec3(m_min.x, m_max.y, m_max.z);
    dir1 = glm::ivec3(0, 1, 0);
    dir2 = glm::ivec3(m_max.x - m_min.x, 0, 0);
  }
  else if (direction == glm::ivec3(0, -1, -1))
  {
    origin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(0, -1, -1);
    dir1 = glm::ivec3(0, 1, 0);
    dir2 = glm::ivec3(m_max.x - m_min.x, 0, 0);
  }
  else if (direction == glm::ivec3(0, 1, -1))
  {
    origin = glm::ivec3(m_min.x, m_max.y, m_min.z) + glm::ivec3(0, 0, -1);
    dir1 = glm::ivec3(0, 1, 0);
    dir2 = glm::ivec3(m_max.x - m_min.x, 0, 0);
  }
  else if (direction == glm::ivec3(0, -1, 1))
  {
    origin = glm::ivec3(m_min.x, m_min.y, m_max.z) + glm::ivec3(0, -1, 0);
    dir1 = glm::ivec3(0, 1, 0);
    dir2 = glm::ivec3(m_max.x - m_min.x, 0, 0);
  }
  // 8 corners
  else if (direction == glm::ivec3(1, 1, 1))
  {
    origin = glm::ivec3(m_max.x, m_max.y, m_max.z);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  }
  else if (direction == glm::ivec3(-1, -1, -1))
  {
    origin = glm::ivec3(m_min.x, m_min.y, m_min.z) + glm::ivec3(-1, -1, -1);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  }
  else if (direction == glm::ivec3(-1, 1, 1))
  {
    origin = glm::ivec3(m_min.x, m_max.y, m_max.z) + glm::ivec3(-1, 0, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  }
  else if (direction == glm::ivec3(1, -1, -1))
  {
    origin = glm::ivec3(m_max.x, m_min.y, m_min.z) + glm::ivec3(0, -1, -1);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  }
  else if (direction == glm::ivec3(1, -1, 1))
  {
    origin = glm::ivec3(m_max.x, m_min.y, m_max.z) + glm::ivec3(0, -1, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  }
  else if (direction == glm::ivec3(-1, 1, -1))
  {
    origin = glm::ivec3(m_min.x, m_max.y, m_min.z) + glm::ivec3(-1, 0, -1);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  }
  else if (direction == glm::ivec3(1, 1, -1))
  {
    origin = glm::ivec3(m_max.x, m_max.y, m_min.z) + glm::ivec3(0, 0, -1);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  }
  else if (direction == glm::ivec3(-1, -1, 1))
  {
    origin = glm::ivec3(m_min.x, m_min.y, m_max.z) + glm::ivec3(-1, -1, 0);
    dir1 = glm::ivec3(1, 0, 0);
    dir2 = glm::ivec3(0, 1, 0);
  }
  else
  {
    throw std::out_of_range("Unknown halo direction vector");
  }
  int n1 = abs(dir1.x) + abs(dir1.y) + abs(dir1.z);
  int n2 = abs(dir2.x) + abs(dir2.y) + abs(dir2.z);
  glm::ivec3 e1 = dir1 / n1;
  glm::ivec3 e2 = dir2 / n2;
  for (int i1 = 0; i1 < n1; i1++)
  {
    for (int i2 = 0; i2 < n2; i2++)
    {
      glm::ivec3 halo = origin + e1 * i1 + e2 * i2;
      haloPoints->push_back(halo);
    }
  }
}

void Topology::buildMesh()
{
  for (int i = 0; i < getNumPartitions(); i++)
  {
    Partition *partition = m_partitions[i];
    osg::Vec4 color;
    {
      float cx = partition->getMin().x + partition->getNx() * 0.5;
      float cy = partition->getMin().y + partition->getNy() * 0.5;
      float cz = partition->getMin().z + partition->getNz() * 0.5;
      osg::Vec3d center(cx, cy, cz);
      osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(new osg::Box(center, partition->getNx(), partition->getNy(), partition->getNz()));

      color = m_colorSet->getColor(i + 2);
      sd->setColor(osg::Vec4f(color.r(), color.g(), color.b(), color.a()));

      osg::ref_ptr<osg::Geode> geode = new osg::Geode();
      geode->addDrawable(sd);
      // osg::ref_ptr<osg::StateSet> stateset = geode->getOrCreateStateSet();
      // osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
      // polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
      // stateset->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
      // osg::Material *material = new osg::Material;
      // stateset->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
      // stateset->setAttributeAndModes(material, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

      m_root->addChild(geode);
    }
    for (std::pair<glm::ivec3, Partition *> neigbour : partition->m_neighbours)
    {
      std::vector<glm::ivec3> haloPoints;
      partition->getHalo(neigbour.first, &haloPoints);
      for (glm::ivec3 haloPoint : haloPoints)
      {
        osg::Vec3d center(haloPoint.x + 0.5, haloPoint.y + 0.5, haloPoint.z + 0.5);
        osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(new osg::Box(center, 1, 1, 1));
        sd->setColor(osg::Vec4f(color.r(), color.g(), color.b(), color.a()));

        osg::ref_ptr<osg::Geode> geode = new osg::Geode();
        geode->addDrawable(sd);
        osg::ref_ptr<osg::StateSet> stateset = geode->getOrCreateStateSet();
        osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
        polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
        stateset->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
        osg::Material *material = new osg::Material;
        stateset->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
        stateset->setAttributeAndModes(material, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

        m_root->addChild(geode);
      }
    }
  }
}

Topology::Topology(unsigned int latticeSizeX,
                   unsigned int latticeSizeY,
                   unsigned int latticeSizeZ,
                   unsigned int subdivisions)
    : m_colorSet(new ColorSet()),
      m_root(new osg::Group()),
      m_partitionCount(glm::ivec3(1, 1, 1)),
      m_latticeSize(glm::ivec3(latticeSizeX, latticeSizeY, latticeSizeZ))
{
  Partition *partition = new Partition(glm::ivec3(0, 0, 0), m_latticeSize);
  m_partitions.push_back(partition);
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
  for (int x = 0; x < getNumPartitionsX(); x++)
    for (int y = 0; y < getNumPartitionsY(); y++)
      for (int z = 0; z < getNumPartitionsZ(); z++)
      {
        glm::ivec3 position(x, y, z);
        Partition *partition = getPartition(position);
        for (glm::ivec3 haloDirection : haloDirections)
        {
          glm::ivec3 neighborPos = position + haloDirection;
          // Periodic
          neighborPos.x = (neighborPos.x == getNumPartitionsX()) ? 0 : neighborPos.x;
          neighborPos.x = (neighborPos.x == -1) ? getNumPartitionsX() - 1 : neighborPos.x;
          neighborPos.y = (neighborPos.y == getNumPartitionsY()) ? 0 : neighborPos.y;
          neighborPos.y = (neighborPos.y == -1) ? getNumPartitionsY() - 1 : neighborPos.y;
          neighborPos.z = (neighborPos.z == getNumPartitionsZ()) ? 0 : neighborPos.z;
          neighborPos.z = (neighborPos.z == -1) ? getNumPartitionsZ() - 1 : neighborPos.z;
          Partition *neighbour = getPartition(neighborPos);
          partition->m_neighbours[haloDirection] = neighbour;
        }
      }
}
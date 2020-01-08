#include "VoxelGeometry.hpp"

VoxelGeometry::VoxelGeometry()
    : m_voxelTypeCounter(1),
      m_incompatible(0),
      m_voxNameMap(),
      m_nameQuadMap() {
  BoundaryCondition empty;
  m_voxelArray = std::make_shared<VoxelArray>(0, 0, 0);
  m_bcsArray = std::make_shared<BoundaryConditions>();
  m_bcsArray->push_back(empty);
  m_sensorArray = std::make_shared<VoxelVolumeArray>();
}

VoxelGeometry::VoxelGeometry(const int nx, const int ny, const int nz)
    : m_voxelTypeCounter(1),
      m_incompatible(0),
      m_voxNameMap(),
      m_nameQuadMap() {
  BoundaryCondition empty;
  m_voxelArray = std::make_shared<VoxelArray>(nx, ny, nz);
  m_voxelArray->allocate();
  m_bcsArray = std::make_shared<BoundaryConditions>();
  m_bcsArray->push_back(empty);
  m_sensorArray = std::make_shared<VoxelVolumeArray>();
}

void VoxelGeometry::storeType(BoundaryCondition* bc,
                              const std::string& geoName) {
  if (bc->m_type == VoxelType::Enum::FLUID) {
    bc->m_id = VoxelType::Enum::FLUID;
  } else if (bc->m_type == VoxelType::Enum::EMPTY) {
    bc->m_id = VoxelType::Enum::EMPTY;
  } else {
    std::size_t hashKey = std::hash<BoundaryCondition>{}(*bc, geoName);
    if (m_types.find(hashKey) == m_types.end()) {
      // Not found, combination of boundary condition and geometry name
      bc->m_id = m_voxelTypeCounter++;
      m_bcsArray->push_back(*bc);
      m_types[hashKey] = *bc;
      m_voxNameMap.resize(bc->m_id + 1);
      m_voxNameMap.at(bc->m_id).insert(geoName);
    } else {
      // Found combination
      bc->m_id = m_types[hashKey].m_id;
    }
    // std::unordered_set<std::string, std::hash<std::string>> myset =
    //     m_voxNameMap[bc->m_id];
    // for (auto it = myset.begin(); it != myset.end(); ++it)
    //   std::cout << " " << *it;
    // std::cout << std::endl;
  }
}

void VoxelGeometry::set(Eigen::Vector3i p,
                        BoundaryCondition bc,
                        NodeMode::Enum mode,
                        std::string name) {
  if (get(p) == VoxelType::Enum::EMPTY || get(p) == VoxelType::Enum::FLUID) {
    // Replacing empty voxel
    set(p, bc.m_id);

  } else if (mode == NodeMode::Enum::OVERWRITE) {
    // Overwrite whatever type was there
    set(p, bc.m_id);

  } else if (mode == NodeMode::Enum::INTERSECT) {
    // There is a boundary already
    voxel_t vox1 = get(p);
    BoundaryCondition oldBc = m_bcsArray->at(vox1);

    std::size_t newHashKey = std::hash<BoundaryCondition>{}(bc);
    std::size_t oldHashKey = std::hash<BoundaryCondition>{}(oldBc);

    if (newHashKey != oldHashKey) {
      // normal of the exiting voxel
      Eigen::Vector3i n1 = oldBc.m_normal;
      // normal of the new boundary
      Eigen::Vector3i n2 = bc.m_normal;
      // build a new vector, sum of the two vectors
      Eigen::Vector3i n = n1 + n2;
      // if the boundaries are opposite, they cannot be compatible, so
      // overwrite with the new boundary
      if (n1.x() == -n2.x() && n1.y() == -n2.y() && n1.z() == -n2.z()) n = n2;
      // TODO(this suppose they have the same boundary type)
      if (bc.m_type != oldBc.m_type) m_incompatible++;

      BoundaryCondition mergeBc = bc;
      mergeBc.m_normal = n;
      storeType(&mergeBc, name);
      set(p, mergeBc.m_id);
    }

  } else if (mode == NodeMode::Enum::FILL) {
    // Not empty, do nothing
  }
}

std::unordered_set<voxel_t> VoxelGeometry::getVoxelsByName(std::string name) {
  std::unordered_set<VoxelQuad> quads = m_nameQuadMap.at(name);
  std::unordered_set<voxel_t> voxIds;
  for (const VoxelQuad quad : quads) {
    voxIds.insert(quad.m_bc.m_id);
    for (BoundaryCondition bc : quad.m_intersectingBcs) voxIds.insert(bc.m_id);
  }
  return voxIds;
}

std::vector<std::string> VoxelGeometry::getGeometryNames() {
  std::vector<std::string> names;
  names.reserve(m_nameQuadMap.size());
  for (std::unordered_map<
           std::string,
           std::unordered_set<VoxelQuad, std::hash<VoxelQuad>>>::iterator it =
           m_nameQuadMap.begin();
       it != m_nameQuadMap.end(); ++it)
    names.push_back(it->first);
  std::sort(names.begin(), names.end(), std::less<std::string>());
  return names;
}

std::unordered_map<Eigen::Vector3i, std::string> VoxelGeometry::getLabels() {
  std::unordered_map<Eigen::Vector3i, std::string> labels;
  for (std::pair<std::string, std::unordered_set<VoxelQuad>> element :
       m_nameQuadMap) {
    std::string name = element.first;
    if (!name.compare(DEFAULT_GEOMETRY_NAME)) continue;
    for (std::unordered_set<VoxelQuad>::iterator itr = element.second.begin();
         itr != element.second.end(); ++itr) {
      Eigen::Vector3i origin = (*itr).m_voxOrigin;
      Eigen::Vector3i dir1 = (*itr).m_voxDir1;
      Eigen::Vector3i dir2 = (*itr).m_voxDir2;
      Eigen::Vector3i pos = origin + dir1 / 2 + dir2 / 2;
      labels[Eigen::Vector3i(pos.x(), pos.y(), pos.z())] = name;
      break;
    }
  }
  return labels;
}

#include "VoxelGeometry.hpp"

VoxelGeometry::VoxelGeometry()
    : m_nx(0), m_ny(0), m_nz(0), m_voxelTypeCounter(1) {
  BoundaryCondition empty;
  m_voxelArray = std::make_shared<VoxelArray>(0, 0, 0);
  m_bcsArray = std::make_shared<BoundaryConditions>();
  m_bcsArray->push_back(empty);
  m_sensorArray = std::make_shared<VoxelVolumeArray>();
}

VoxelGeometry::VoxelGeometry(const int nx, const int ny, const int nz)
    : m_nx(nx), m_ny(ny), m_nz(nz), m_voxelTypeCounter(1) {
  BoundaryCondition empty;
  m_voxelArray = std::make_shared<VoxelArray>(nx, ny, nz);
  m_voxelArray->allocate();
  m_bcsArray = std::make_shared<BoundaryConditions>();
  m_bcsArray->push_back(empty);
  m_sensorArray = std::make_shared<VoxelVolumeArray>();
}

voxel_t VoxelGeometry::storeType(BoundaryCondition *bc,
                                 const std::string &geoName) {
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
    } else {
      // Found combination
      bc->m_id = m_types[hashKey].m_id;
    }
  }
  return bc->m_id;
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

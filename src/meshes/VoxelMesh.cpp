#include "VoxelMesh.hpp"

// Constructor with an existing voxel array
VoxelMesh::VoxelMesh(VoxelArray *voxels)
    : osg::Geometry(),
      m_transform(new osg::PositionAttitudeTransform()),
      m_voxels(voxels),
      m_polyMode(osg::PolygonMode::Mode::FILL) {
  m_colorSet = new ColorSet();

  m_arrayOrig = new MeshArray();
  m_arrayTmp1 = new MeshArray();
  m_arrayTmp2 = new MeshArray();

  setUseVertexBufferObjects(true);
  addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));

  build(VoxelMeshType::REDUCED);
}

// Copy constructor
VoxelMesh::VoxelMesh(const VoxelMesh &other)
    : osg::Geometry(),
      m_transform(new osg::PositionAttitudeTransform(*other.m_transform)),
      m_voxels(other.m_voxels),
      m_polyMode(other.m_polyMode),
      m_colorSet(other.m_colorSet) {
  m_arrayOrig = new MeshArray();
  m_arrayTmp1 = new MeshArray();
  m_arrayTmp2 = new MeshArray();

  m_arrayOrig->insert(other.m_arrayOrig);
  m_arrayTmp1->insert(other.m_arrayTmp1);
  m_arrayTmp2->insert(other.m_arrayTmp2);

  setUseVertexBufferObjects(true);
  addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));

  bind(m_arrayTmp1);
}

void VoxelMesh::setPolygonMode(osg::PolygonMode::Mode mode) {
  m_polyMode = mode;
  osg::ref_ptr<osg::StateSet> stateset = getOrCreateStateSet();
  osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
  polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, mode);
  stateset->setAttributeAndModes(
      polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
}

void VoxelMesh::bind(MeshArray *array) {
  array->dirty();

  setVertexArray(array->m_vertices);
  setColorArray(array->m_colors);
  setNormalArray(array->m_normals);
  setTexCoordArray(0, array->m_texCoords);

  setColorBinding(osg::Geometry::BIND_PER_VERTEX);
  setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

  osg::DrawArrays *drawArrays =
      static_cast<osg::DrawArrays *>(getPrimitiveSet(0));
  drawArrays->setCount(array->m_vertices->getNumElements());
  drawArrays->dirty();

  osg::ref_ptr<osg::StateSet> stateset = getOrCreateStateSet();

  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
  stateset->setMode(GL_LIGHTING,
                    osg::StateAttribute::ON | osg::StateAttribute::PROTECTED);

  osg::ref_ptr<osg::Material> mat = new osg::Material();
  mat->setAmbient(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 1.0f);
  mat->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.5f);
  mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
                   osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.1f);
  mat->setColorMode(osg::Material::ColorMode::AMBIENT_AND_DIFFUSE);

  stateset->setAttribute(mat.get(), osg::StateAttribute::Values::ON);

  setPolygonMode(m_polyMode);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(this);

  if (m_transform->getNumChildren() > 0)
    m_transform->removeChildren(0, m_transform->getNumChildren());
  m_transform->addChild(geode);
}

void VoxelMesh::crop(osg::Vec3i voxMin, osg::Vec3i voxMax) {
  m_arrayTmp2->clear();
  crop(m_arrayOrig, m_arrayTmp2, voxMin, voxMax);
  bind(m_arrayTmp2);
  MeshArray::swap(m_arrayTmp1, m_arrayTmp2);
}

void VoxelMesh::crop(MeshArray *src, MeshArray *dst, osg::Vec3i voxMin,
                     osg::Vec3i voxMax) {
  for (int i = 0; i < src->m_vertices->getNumElements(); i += 4) {
    osg::Vec3 v1 = src->m_vertices->at(i);
    osg::Vec3 v2 = src->m_vertices->at(i + 1);
    osg::Vec3 v3 = src->m_vertices->at(i + 2);
    osg::Vec3 v4 = src->m_vertices->at(i + 3);
    if (limitPolygon(&v1, &v2, &v3, &v4, voxMin,
                     voxMax + osg::Vec3i(1, 1, 1))) {
      dst->m_vertices->push_back(v1);
      dst->m_vertices->push_back(v2);
      dst->m_vertices->push_back(v3);
      dst->m_vertices->push_back(v4);
      dst->m_colors->push_back(src->m_colors->at(i));
      dst->m_colors->push_back(src->m_colors->at(i + 1));
      dst->m_colors->push_back(src->m_colors->at(i + 2));
      dst->m_colors->push_back(src->m_colors->at(i + 3));
      dst->m_normals->push_back(src->m_normals->at(i));
      dst->m_normals->push_back(src->m_normals->at(i + 1));
      dst->m_normals->push_back(src->m_normals->at(i + 2));
      dst->m_normals->push_back(src->m_normals->at(i + 3));
      // TODO(Texture coords not changed)
      dst->m_texCoords->push_back(src->m_texCoords->at(i));
      dst->m_texCoords->push_back(src->m_texCoords->at(i + 1));
      dst->m_texCoords->push_back(src->m_texCoords->at(i + 2));
      dst->m_texCoords->push_back(src->m_texCoords->at(i + 3));
    }
  }
}

bool VoxelMesh::limitPolygon(osg::Vec3 *v1, osg::Vec3 *v2, osg::Vec3 *v3,
                             osg::Vec3 *v4, osg::Vec3i min, osg::Vec3i max) {
  osg::Vec3 *vecs[4] = {v1, v2, v3, v4};
  if (v1->x() < min.x() && v2->x() < min.x() && v3->x() < min.x() &&
      v4->x() < min.x())
    return false;
  if (v1->y() < min.y() && v2->y() < min.y() && v3->y() < min.y() &&
      v4->y() < min.y())
    return false;
  if (v1->z() < min.z() && v2->z() < min.z() && v3->z() < min.z() &&
      v4->z() < min.z())
    return false;
  if (v1->x() > max.x() && v2->x() > max.x() && v3->x() > max.x() &&
      v4->x() > max.x())
    return false;
  if (v1->y() > max.y() && v2->y() > max.y() && v3->y() > max.y() &&
      v4->y() > max.y())
    return false;
  if (v1->z() > max.z() && v2->z() > max.z() && v3->z() > max.z() &&
      v4->z() > max.z())
    return false;
  for (int i = 0; i < 4; i++) {
    osg::Vec3 *v = vecs[i];
    v->x() = std::max(v->x(), static_cast<float>(min.x()));
    v->y() = std::max(v->y(), static_cast<float>(min.y()));
    v->z() = std::max(v->z(), static_cast<float>(min.z()));
    v->x() = std::min(v->x(), static_cast<float>(max.x()));
    v->y() = std::min(v->y(), static_cast<float>(max.y()));
    v->z() = std::min(v->z(), static_cast<float>(max.z()));
  }
  return true;
}

void VoxelMesh::build(VoxelMeshType::Enum type) {
  m_arrayOrig->clear();
  m_arrayTmp1->clear();

  switch (type) {
    case VoxelMeshType::FULL:
      buildMeshFull(m_arrayOrig);
      break;
    case VoxelMeshType::REDUCED:
      buildMeshReduced(m_arrayOrig);
      break;
    default:
      return;
  }
  m_arrayTmp1->insert(m_arrayOrig);
  bind(m_arrayTmp1);

  std::cout << "Voxel mesh: " << m_arrayTmp1->m_vertices->getNumElements()
            << " vertices" << std::endl;
}

static void reduce(std::vector<MeshArray *> v, int begin, int end) {
  if (end - begin == 1) return;
  int pivot = (begin + end) / 2;
#pragma omp task
  reduce(v, begin, pivot);
#pragma omp task
  reduce(v, pivot, end);
#pragma omp taskwait
  v.at(begin)->m_vertices->insert(v.at(begin)->m_vertices->end(),
                                  v.at(pivot)->m_vertices->begin(),
                                  v.at(pivot)->m_vertices->end());
  v.at(begin)->m_normals->insert(v.at(begin)->m_normals->end(),
                                 v.at(pivot)->m_normals->begin(),
                                 v.at(pivot)->m_normals->end());
  v.at(begin)->m_colors->insert(v.at(begin)->m_colors->end(),
                                v.at(pivot)->m_colors->begin(),
                                v.at(pivot)->m_colors->end());
  v.at(begin)->m_texCoords->insert(v.at(begin)->m_texCoords->end(),
                                   v.at(pivot)->m_texCoords->begin(),
                                   v.at(pivot)->m_texCoords->end());
}

void VoxelMesh::buildMeshReduced(MeshArray *array) {
  // Slice the voxel array into number of threads and merge them later.
  // Will decrease build time, but increase triangle count slightly...
  const int numSlices = omp_get_num_procs();

  // Slice axis
  D3Q7::Enum axis = D3Q7::Y_AXIS_POS;

  std::vector<MeshArray *> arrayPtr(numSlices);
  const int dims[3] = {static_cast<int>(m_voxels->getSizeX()),
                       static_cast<int>(m_voxels->getSizeY()),
                       static_cast<int>(m_voxels->getSizeZ())};
  const int d[3] = {static_cast<int>(std::ceil(1.0 * dims[0] / numSlices)),
                    static_cast<int>(std::ceil(1.0 * dims[1] / numSlices)),
                    static_cast<int>(std::ceil(1.0 * dims[2] / numSlices))};

#pragma omp parallel num_threads(numSlices)
  {
    const int id = omp_get_thread_num();
    arrayPtr.at(id) = new MeshArray();
    int min[3] = {0, 0, 0};
    int max[3] = {dims[0], dims[1], dims[2]};
    switch (axis) {
      case D3Q7::X_AXIS_POS:
        min[0] = d[0] * id;
        if (id < numSlices - 1) max[0] = min[0] + d[0];
        break;
      case D3Q7::Y_AXIS_POS:
        min[1] = d[1] * id;
        if (id < numSlices - 1) max[1] = min[1] + d[1];
        break;
      case D3Q7::Z_AXIS_POS:
        min[2] = d[2] * id;
        if (id < numSlices - 1) max[2] = min[2] + d[2];
        break;
      default:
        break;
    }
    buildMeshReduced(arrayPtr.at(id), min, max);
  }
  reduce(arrayPtr, 0, numSlices);

  array->m_vertices = arrayPtr.at(0)->m_vertices;
  array->m_colors = arrayPtr.at(0)->m_colors;
  array->m_normals = arrayPtr.at(0)->m_normals;
  array->m_texCoords = arrayPtr.at(0)->m_texCoords;

  for (int i = 1; i < numSlices; i++) delete arrayPtr.at(i);
}

void VoxelMesh::buildMeshReduced(MeshArray *array, int min[3], int max[3]) {
  int dims[3] = {max[0] - min[0], max[1] - min[1], max[2] - min[2]};

  for (bool backFace = true, b = false; b != backFace;
       backFace = backFace && b, b = !b) {
    // Sweep over 3-axes
    for (int d = 0; d < 3; ++d) {
      int i, j, k, l, w, h, u = (d + 1) % 3, v = (d + 2) % 3;
      int x[3] = {0, 0, 0};
      int q[3] = {0, 0, 0};
      std::vector<int> mask((dims[u] + 1) * (dims[v] + 1));
      q[d] = 1;
      for (x[d] = -1; x[d] < dims[d];) {
        // Compute mask
        int n = 0;
        for (x[v] = 0; x[v] < dims[v]; ++x[v])
          for (x[u] = 0; x[u] < dims[u]; ++x[u], ++n) {
            voxel va =
                (0 <= x[d] ? m_voxels->getVoxelReadOnly(
                                 x[0] + min[0], x[1] + min[1], x[2] + min[2])
                           : 0);
            voxel vb = (x[d] < dims[d] - 1
                            ? m_voxels->getVoxelReadOnly(x[0] + q[0] + min[0],
                                                         x[1] + q[1] + min[1],
                                                         x[2] + q[2] + min[2])
                            : 0);
            if (va == vb) {
              mask.at(n) = 0;
            } else if (backFace) {
              mask.at(n) = vb;
            } else {
              mask.at(n) = va;
            }
          }
        // Increment x[d]
        ++x[d];
        // Generate mesh for mask using lexicographic ordering
        n = 0;
        for (j = 0; j < dims[v]; ++j)
          for (i = 0; i < dims[u];) {
            voxel c = mask.at(n);
            if (c) {
              // Compute width
              for (w = 1; c == mask.at(n + w) && i + w < dims[u]; ++w) {
              }
              // Compute height (this is slightly awkward
              bool done = false;
              for (h = 1; j + h < dims[v]; ++h) {
                for (k = 0; k < w; ++k) {
                  if (c != mask.at(n + k + h * dims[u])) {
                    done = true;
                    break;
                  }
                }
                if (done) break;
              }
              // Add quad
              x[u] = i;
              x[v] = j;
              int p[3] = {x[0] + min[0], x[1] + min[1], x[2] + min[2]};
              int du[3] = {0, 0, 0};
              int dv[3] = {0, 0, 0};
              if (c > 0) {
                dv[v] = h;
                du[u] = w;
                osg::Vec3 v1(p[0], p[1], p[2]);
                osg::Vec3 v2(p[0] + du[0], p[1] + du[1], p[2] + du[2]);
                osg::Vec3 v3(p[0] + du[0] + dv[0], p[1] + du[1] + dv[1],
                             p[2] + du[2] + dv[2]);
                osg::Vec3 v4(p[0] + dv[0], p[1] + dv[1], p[2] + dv[2]);

                osg::Vec4 color = m_colorSet->getColor(c);

                osg::Vec3 normal(0, 0, 0);

                if (d == 0 && backFace)
                  normal = osg::Vec3(-1, 0, 0);
                else if (d == 0 && !backFace)
                  normal = osg::Vec3(1, 0, 0);
                else if (d == 1 && backFace)
                  normal = osg::Vec3(0, -1, 0);
                else if (d == 1 && !backFace)
                  normal = osg::Vec3(0, 1, 0);
                else if (d == 2 && backFace)
                  normal = osg::Vec3(0, 0, -1);
                else if (d == 2 && !backFace)
                  normal = osg::Vec3(0, 0, 1);

                array->m_vertices->push_back(v1);
                array->m_vertices->push_back(v2);
                array->m_vertices->push_back(v3);
                array->m_vertices->push_back(v4);

                array->m_colors->push_back(color);
                array->m_colors->push_back(color);
                array->m_colors->push_back(color);
                array->m_colors->push_back(color);

                array->m_normals->push_back(normal);
                array->m_normals->push_back(normal);
                array->m_normals->push_back(normal);
                array->m_normals->push_back(normal);

                array->m_texCoords->push_back(osg::Vec2(0, 0));
                array->m_texCoords->push_back(osg::Vec2(w, 0));
                array->m_texCoords->push_back(osg::Vec2(w, h));
                array->m_texCoords->push_back(osg::Vec2(0, h));
              }

              // Zero-out mask
              for (l = 0; l < h; ++l)
                for (k = 0; k < w; ++k) mask.at(n + k + l * dims[u]) = 0;

              // Increment counters and continue
              i += w;
              n += w;
            } else {
              ++i;
              ++n;
            }
          }
      }
    }
  }
}

void VoxelMesh::buildMeshFull(MeshArray *array) {
  for (int k = 0; k < static_cast<int>(m_voxels->getSizeZ()); ++k) {
    for (int j = 0; j < static_cast<int>(m_voxels->getSizeY()); ++j) {
      for (int i = 0; i < static_cast<int>(m_voxels->getSizeX()); ++i) {
        if (!m_voxels->isEmpty(i, j, k)) {
          voxel v = m_voxels->getVoxelReadOnly(i, j, k);
          osg::Vec4 color = m_colorSet->getColor(v);

          if (m_voxels->isEmpty(i + 1, j, k)) {
            osg::Vec3 normal(1.0f, 0.0f, 0.0f);
            array->m_vertices->push_back(osg::Vec3(i + 1, j, k));
            array->m_vertices->push_back(osg::Vec3(i + 1, j + 1, k));
            array->m_vertices->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            array->m_vertices->push_back(osg::Vec3(i + 1, j, k + 1));
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
          }
          if (m_voxels->isEmpty(i - 1, j, k)) {
            osg::Vec3 normal(-1.0f, 0.0f, 0.0f);
            array->m_vertices->push_back(osg::Vec3(i, j, k));
            array->m_vertices->push_back(osg::Vec3(i, j + 1, k));
            array->m_vertices->push_back(osg::Vec3(i, j + 1, k + 1));
            array->m_vertices->push_back(osg::Vec3(i, j, k + 1));
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
          }
          if (m_voxels->isEmpty(i, j + 1, k)) {
            osg::Vec3 normal(0.0f, 1.0f, 0.0f);
            array->m_vertices->push_back(osg::Vec3(i, j + 1, k));
            array->m_vertices->push_back(osg::Vec3(i + 1, j + 1, k));
            array->m_vertices->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            array->m_vertices->push_back(osg::Vec3(i, j + 1, k + 1));
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
          }
          if (m_voxels->isEmpty(i, j - 1, k)) {
            osg::Vec3 normal(0.0f, -1.0f, 0.0f);
            array->m_vertices->push_back(osg::Vec3(i, j, k));
            array->m_vertices->push_back(osg::Vec3(i + 1, j, k));
            array->m_vertices->push_back(osg::Vec3(i + 1, j, k + 1));
            array->m_vertices->push_back(osg::Vec3(i, j, k + 1));
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
          }
          if (m_voxels->isEmpty(i, j, k + 1)) {
            osg::Vec3 normal(0.0f, 0.0f, 1.0f);
            array->m_vertices->push_back(osg::Vec3(i, j, k + 1));
            array->m_vertices->push_back(osg::Vec3(i + 1, j, k + 1));
            array->m_vertices->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            array->m_vertices->push_back(osg::Vec3(i, j + 1, k + 1));
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
          }
          if (m_voxels->isEmpty(i, j, k - 1)) {
            osg::Vec3 normal(0.0f, 0.0f, -1.0f);
            array->m_vertices->push_back(osg::Vec3(i, j, k));
            array->m_vertices->push_back(osg::Vec3(i + 1, j, k));
            array->m_vertices->push_back(osg::Vec3(i + 1, j + 1, k));
            array->m_vertices->push_back(osg::Vec3(i, j + 1, k));
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_normals->push_back(normal);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_colors->push_back(color);
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
            array->m_texCoords->push_back(osg::Vec2(0, 0));
          }
        }
      }
    }
  }
}

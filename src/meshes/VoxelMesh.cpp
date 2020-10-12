#include "VoxelMesh.hpp"

// Constructor with an existing voxel array
VoxelMesh::VoxelMesh(std::shared_ptr<VoxelArray> voxels)
    : osg::Geode(),
      m_geo(new osg::Geometry()),
      m_size(voxels->getSizeX(), voxels->getSizeY(), voxels->getSizeZ()),
      m_polyMode(osg::PolygonMode::Mode::FILL) {
  m_arrayOrig = new MeshArray();
  m_arrayTmp1 = new MeshArray();
  m_arrayTmp2 = new MeshArray();

  m_geo->setUseVertexBufferObjects(true);
  m_geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));

  build(voxels);
}

VoxelMesh::VoxelMesh(const std::string filePath, osg::Vec3i size)
    : osg::Geode(),
      m_geo(new osg::Geometry()),
      m_size(size),
      m_polyMode(osg::PolygonMode::Mode::FILL) {
  osg::ref_ptr<osg::Node> node = osgDB::readNodeFile(filePath);
  osg::ref_ptr<osg::Geometry> geo =
      node->asGeode()->getDrawable(0)->asGeometry();

  m_arrayOrig = new MeshArray(geo);
  m_arrayTmp1 = new MeshArray(geo);
  m_arrayTmp2 = new MeshArray(geo);

  m_geo->setUseVertexBufferObjects(true);
  m_geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));

  bind(m_arrayTmp1);
}

// Copy constructor
VoxelMesh::VoxelMesh(const VoxelMesh& other)
    : osg::Geode(),
      m_geo(new osg::Geometry()),
      m_size(other.m_size),
      m_colorSet(other.m_colorSet),
      m_polyMode(other.m_polyMode) {
  m_arrayOrig = new MeshArray();
  m_arrayTmp1 = new MeshArray();
  m_arrayTmp2 = new MeshArray();

  m_arrayOrig->insert(other.m_arrayOrig);
  m_arrayTmp1->insert(other.m_arrayTmp1);
  m_arrayTmp2->insert(other.m_arrayTmp2);

  m_geo->setUseVertexBufferObjects(true);
  m_geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));

  bind(m_arrayTmp1);
}

void VoxelMesh::setPolygonMode(osg::PolygonMode::Mode mode) {
  m_polyMode = mode;
  osg::ref_ptr<osg::StateSet> stateset = m_geo->getOrCreateStateSet();
  osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
  polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, mode);
  stateset->setAttributeAndModes(polymode, osg::StateAttribute::ON);
}

void VoxelMesh::bind(MeshArray* array) {
  array->dirty();

  m_geo->setVertexArray(array->m_vertices);
  m_geo->setColorArray(array->m_colors, osg::Array::BIND_PER_VERTEX);
  m_geo->setNormalArray(array->m_normals, osg::Array::BIND_PER_VERTEX);
  m_geo->setTexCoordArray(0, array->m_texCoords);

  osg::DrawArrays* drawArrays =
      static_cast<osg::DrawArrays*>(m_geo->getPrimitiveSet(0));
  drawArrays->setCount(array->m_vertices->getNumElements());
  drawArrays->dirty();

  osg::ref_ptr<osg::StateSet> stateset = m_geo->getOrCreateStateSet();

  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
  stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);

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

  addDrawable(m_geo);
}

void VoxelMesh::crop(osg::Vec3i voxMin, osg::Vec3i voxMax) {
  m_arrayTmp2->clear();
  crop(m_arrayOrig, m_arrayTmp2, voxMin, voxMax);
  bind(m_arrayTmp2);
  MeshArray::swap(m_arrayTmp1, m_arrayTmp2);
}

void VoxelMesh::crop(MeshArray* src,
                     MeshArray* dst,
                     osg::Vec3i voxMin,
                     osg::Vec3i voxMax) {
  for (unsigned int i = 0; i < src->m_vertices->getNumElements(); i += 4) {
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

bool VoxelMesh::limitPolygon(osg::Vec3* v1,
                             osg::Vec3* v2,
                             osg::Vec3* v3,
                             osg::Vec3* v4,
                             osg::Vec3i min,
                             osg::Vec3i max) {
  osg::Vec3* vecs[4] = {v1, v2, v3, v4};
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
    osg::Vec3* v = vecs[i];
    v->x() = std::max(v->x(), static_cast<float>(min.x()));
    v->y() = std::max(v->y(), static_cast<float>(min.y()));
    v->z() = std::max(v->z(), static_cast<float>(min.z()));
    v->x() = std::min(v->x(), static_cast<float>(max.x()));
    v->y() = std::min(v->y(), static_cast<float>(max.y()));
    v->z() = std::min(v->z(), static_cast<float>(max.z()));
  }
  return true;
}

void VoxelMesh::build(std::shared_ptr<VoxelArray> voxels) {
  m_arrayOrig->clear();
  m_arrayTmp1->clear();

  buildMeshReduced(voxels, m_arrayOrig);

  m_arrayTmp1->insert(m_arrayOrig);
  bind(m_arrayTmp1);

  std::cout << "Built voxel mesh with " << m_arrayTmp1->size() / 4 << " quads"
            << std::endl;
}

static void reduce(std::vector<MeshArray*> v, int begin, int end) {
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

void VoxelMesh::buildMeshReduced(std::shared_ptr<VoxelArray> voxels,
                                 MeshArray* array) {
  enum StripeType { NONE, ERASE, MERGE_X, MERGE_Y };
  struct Stripe {
    int id;
    StripeType type;
  } StripeDefault = {-1, NONE};

  // One mesh per thread (leave one for other tasks)
  const unsigned int numSlices = omp_get_num_procs() - 1;
  // Slice axis
  D3Q4::Enum axis = D3Q4::Y_AXIS;

  std::vector<MeshArray*> meshArrays(numSlices);
  std::vector<std::vector<Stripe>*> stripeArrays(numSlices);
  const int exts[3] = {static_cast<int>(voxels->getSizeX()),
                       static_cast<int>(voxels->getSizeY()),
                       static_cast<int>(voxels->getSizeZ())};
  const double d[3] = {1.0 * exts[0] / numSlices, 1.0 * exts[1] / numSlices,
                       1.0 * exts[2] / numSlices};

#pragma omp parallel num_threads(numSlices)
  {
    const unsigned int id = omp_get_thread_num();

    MeshArray* myMeshArray = new MeshArray();
    meshArrays.at(id) = myMeshArray;
    int min[3] = {0, 0, 0};
    int max[3] = {exts[0], exts[1], exts[2]};

    // Calculate which part of the mesh to generate on this thread
    double minf, maxf;
    switch (axis) {
      case D3Q4::X_AXIS:
        minf = std::floor(d[0] * id);
        maxf = std::floor(d[0] * (id + 1));
        min[0] = static_cast<int>(minf);
        if (id < numSlices - 1) max[0] = static_cast<int>(maxf);
        break;
      case D3Q4::Y_AXIS:
        minf = std::floor(d[1] * id);
        maxf = std::floor(d[1] * (id + 1));
        min[1] = static_cast<int>(minf);
        if (id < numSlices - 1) max[1] = static_cast<int>(maxf);
        break;
      case D3Q4::Z_AXIS:
        minf = std::floor(d[2] * id);
        maxf = std::floor(d[2] * (id + 1));
        min[2] = static_cast<int>(minf);
        if (id < numSlices - 1) max[2] = static_cast<int>(maxf);
        break;
      default:
        break;
    }
    // Build part of the mesh
    buildMeshReduced(voxels, myMeshArray, min, max);
    // Here we will collect data on which quads to merge with in next array.
    // This will create a table of which meshes to merge with others
    std::vector<Stripe>* myStripes =
        new std::vector<Stripe>(myMeshArray->size() / 4, StripeDefault);
    stripeArrays.at(id) = myStripes;

#pragma omp barrier
    // When all threads have finished building its mesh part,
    // check the next adjacent mesh for quads to merge or erase
    if (id < numSlices - 1) {
      MeshArray* nextMeshArray = meshArrays.at(id + 1);
      if (myMeshArray->size() > 0) {
        for (unsigned int i = 0; i < myMeshArray->size() - 4; i += 4) {
          // For each quad created in this mesh...
          osg::Vec3& v1 = myMeshArray->m_vertices->at(i);
          osg::Vec3& v2 = myMeshArray->m_vertices->at(i + 1);
          osg::Vec3& v3 = myMeshArray->m_vertices->at(i + 2);
          osg::Vec3& v4 = myMeshArray->m_vertices->at(i + 3);
          osg::Vec3& n1 = myMeshArray->m_normals->at(i);
          osg::Vec4& c1 = myMeshArray->m_colors->at(i);
          if (nextMeshArray->size() > 0) {
            for (unsigned int j = 0; j < nextMeshArray->size() - 4; j += 4) {
              // ... check the adjacent mesh for merge candidates
              osg::Vec3& u1 = nextMeshArray->m_vertices->at(j);
              osg::Vec3& u2 = nextMeshArray->m_vertices->at(j + 1);
              osg::Vec3& u3 = nextMeshArray->m_vertices->at(j + 2);
              osg::Vec3& u4 = nextMeshArray->m_vertices->at(j + 3);
              osg::Vec3& m1 = nextMeshArray->m_normals->at(j);
              osg::Vec4& d1 = nextMeshArray->m_colors->at(j);

              if (v2 == u1 && v3 == u4 && n1 == m1 && c1 == d1) {
                // The quads are of same type and share two edges
                myStripes->at(i / 4) = {j / 4, MERGE_X};
                break;
              }
              if (v3 == u2 && v4 == u1 && n1 == m1 && c1 == d1) {
                // The quads are of same type and share two edges
                myStripes->at(i / 4) = {j / 4, MERGE_Y};
                break;
              }
              if (v1 == u1 && v2 == u2 && v3 == u3 && v4 == u4 && c1 == d1 &&
                  n1 == -m1) {
                // The quads are same type, facing each other and should be
                // removed
                myStripes->at(i / 4) = {j / 4, ERASE};
                break;
              }
            }
          }
        }
      }
    }
  }

  // Loop over each created mesh in order of adjacency
  for (unsigned int sliceIdx = 0; sliceIdx < numSlices - 1; sliceIdx++) {
    MeshArray* myMeshArray = meshArrays.at(sliceIdx);
    std::vector<Stripe>* myStripes = stripeArrays.at(sliceIdx);
    // Parallel loop over each quad's merge data in this mesh
#pragma omp parallel for
    for (size_t i = 0; i < myStripes->size(); i++) {
      Stripe stripe = myStripes->at(i);
      if (stripe.type == NONE) {
        // Nothing to do
        continue;
      } else if (stripe.type == ERASE) {
        // The quad is facing one in the adjacent mesh. Remove both.
        if (stripe.id != -1)
          stripeArrays.at(sliceIdx + 1)->at(stripe.id) = {-1, ERASE};
      } else {
        // The quad should be merged
        // osg::Vec3& v1 = myMeshArray->m_vertices->at(i * 4);
        osg::Vec3& v2 = myMeshArray->m_vertices->at(i * 4 + 1);
        osg::Vec3& v3 = myMeshArray->m_vertices->at(i * 4 + 2);
        osg::Vec3& v4 = myMeshArray->m_vertices->at(i * 4 + 3);
        int nextStripeId = stripe.id;
        StripeType nextStripeType = stripe.type;

        // Traverse the adjacency table of the next meshes until no more
        // quads can be merged into this one
        for (unsigned int nextSliceIdx = sliceIdx + 1; nextSliceIdx < numSlices;
             nextSliceIdx++) {
          std::vector<Stripe>* nextStripes = stripeArrays.at(nextSliceIdx);
          int currentStripeId = nextStripeId;
          Stripe nextStripe = nextStripes->at(currentStripeId);

          if (nextStripe.type == MERGE_X || nextStripe.type == MERGE_Y) {
            // This quad can be merged with the next one. Mark it for removal
            // and continue the traversal
            nextStripeId = nextStripe.id;
            nextStripeType = nextStripe.type;
            nextStripes->at(currentStripeId) = {-1, ERASE};

          } else {
            // This quad cannot be merged with the next one. Stop the traversal.
            MeshArray* nextMeshArray = meshArrays.at(nextSliceIdx);
            // osg::Vec3& u1 = nextMeshArray->m_vertices->at(nextStripeId * 4);
            osg::Vec3& u2 = nextMeshArray->m_vertices->at(nextStripeId * 4 + 1);
            osg::Vec3& u3 = nextMeshArray->m_vertices->at(nextStripeId * 4 + 2);
            osg::Vec3& u4 = nextMeshArray->m_vertices->at(nextStripeId * 4 + 3);
            if (nextStripeType == MERGE_X) {
              v2 = u2;
              v3 = u3;
            } else if (nextStripeType == MERGE_Y) {
              v3 = u3;
              v4 = u4;
            }
            nextStripes->at(currentStripeId) = {-1, ERASE};
            break;
          }
        }
      }
    }
  }

  // Erase any quads marked for removal
#pragma omp parallel num_threads(numSlices)
  {
    const unsigned int id = omp_get_thread_num();
    MeshArray* myMeshArray = meshArrays.at(id);
    std::vector<Stripe>* myStripes = stripeArrays.at(id);
    // Gather the indices of quads to remove
    std::vector<int> deletions;
    for (size_t i = 0; i < myStripes->size(); i++) {
      if (myStripes->at(i).type == ERASE) deletions.push_back(i);
    }
    // Delete quads in descending order of index
    std::sort(deletions.begin(), deletions.end(),
              [](int a, int b) { return a > b; });
    for (size_t i = 0; i < deletions.size(); i++) {
      int index = deletions.at(i);
      myMeshArray->erase(index * 4, index * 4 + 4);
    }
  }
  // Merge the mesh arrays into one
  reduce(meshArrays, 0, numSlices);

  array->m_vertices = meshArrays.at(0)->m_vertices;
  array->m_colors = meshArrays.at(0)->m_colors;
  array->m_normals = meshArrays.at(0)->m_normals;
  array->m_texCoords = meshArrays.at(0)->m_texCoords;

  for (unsigned int i = 1; i < numSlices; i++) delete meshArrays.at(i);
  for (unsigned int i = 0; i < numSlices; i++) delete stripeArrays.at(i);
}

void VoxelMesh::buildMeshReduced(std::shared_ptr<VoxelArray> voxels,
                                 MeshArray* array,
                                 int min[3],
                                 int max[3]) {
  int exts[3] = {max[0] - min[0], max[1] - min[1], max[2] - min[2]};

  for (bool backFace = true, b = false; b != backFace;
       backFace = backFace && b, b = !b) {
    // Sweep over 3-axes
    for (int d = 0; d < 3; ++d) {
      int i, j, k, l, w, h, u = (d + 1) % 3, v = (d + 2) % 3;
      int x[3] = {0, 0, 0};
      int q[3] = {0, 0, 0};
      std::vector<voxel_t> mask((exts[u] + 1) * (exts[v] + 1));
      q[d] = 1;
      for (x[d] = -1; x[d] < exts[d];) {
        // Compute mask
        int n = 0;
        for (x[v] = 0; x[v] < exts[v]; ++x[v])
          for (x[u] = 0; x[u] < exts[u]; ++x[u], ++n) {
            voxel_t va =
                (0 <= x[d] ? voxels->getVoxelReadOnly(
                                 x[0] + min[0], x[1] + min[1], x[2] + min[2])
                           : 0);
            voxel_t vb = (x[d] < exts[d] - 1
                              ? voxels->getVoxelReadOnly(x[0] + q[0] + min[0],
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
        for (j = 0; j < exts[v]; ++j)
          for (i = 0; i < exts[u];) {
            voxel_t c = mask.at(n);
            if (c) {
              // Compute width
              for (w = 1; c == mask.at(n + w) && i + w < exts[u]; ++w) {}
              // Compute height (this is slightly awkward
              bool done = false;
              for (h = 1; j + h < exts[v]; ++h) {
                for (k = 0; k < w; ++k) {
                  if (c != mask.at(n + k + h * exts[u])) {
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

                osg::Vec4 color = m_colorSet.getColor(c);

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
                for (k = 0; k < w; ++k) mask.at(n + k + l * exts[u]) = 0;

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

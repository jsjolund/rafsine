#include "VoxelMesh.hpp"

// Constructor with an existing voxel array
VoxelMesh::VoxelMesh(VoxelArray *voxels)
    : osg::Geometry(),
      m_transform(new osg::PositionAttitudeTransform()),
      m_voxels(voxels),
      m_polyMode(osg::PolygonMode::Mode::FILL) {
  m_colorSet = new ColorSet();

  m_vertexArray = new osg::Vec3Array();
  m_colorArray = new osg::Vec4Array();
  m_normalsArray = new osg::Vec3Array();
  m_texCoordArray = new osg::Vec2Array();

  m_vertexArrayTmp1 = new osg::Vec3Array();
  m_colorArrayTmp1 = new osg::Vec4Array();
  m_normalsArrayTmp1 = new osg::Vec3Array();
  m_texCoordArrayTmp1 = new osg::Vec2Array();

  m_vertexArrayTmp2 = new osg::Vec3Array();
  m_colorArrayTmp2 = new osg::Vec4Array();
  m_normalsArrayTmp2 = new osg::Vec3Array();
  m_texCoordArrayTmp2 = new osg::Vec2Array();

  setUseVertexBufferObjects(true);
  addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));
}

void VoxelMesh::setPolygonMode(osg::PolygonMode::Mode mode) {
  m_polyMode = mode;
  osg::ref_ptr<osg::StateSet> stateset = getOrCreateStateSet();
  osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
  polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, mode);
  stateset->setAttributeAndModes(
      polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
}

void VoxelMesh::bind(osg::ref_ptr<osg::Vec3Array> vertexArray,
                     osg::ref_ptr<osg::Vec4Array> colorArray,
                     osg::ref_ptr<osg::Vec3Array> normalsArray,
                     osg::ref_ptr<osg::Vec2Array> textureArray) {
  vertexArray->dirty();
  colorArray->dirty();
  normalsArray->dirty();
  textureArray->dirty();

  setVertexArray(vertexArray);
  setColorArray(colorArray);
  setNormalArray(normalsArray);
  setTexCoordArray(0, textureArray);

  setColorBinding(osg::Geometry::BIND_PER_VERTEX);
  setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

  osg::DrawArrays *drawArrays =
      static_cast<osg::DrawArrays *>(getPrimitiveSet(0));
  drawArrays->setCount(vertexArray->getNumElements());
  drawArrays->dirty();
  std::cout << vertexArray->getNumElements() << " vertices" << std::endl;

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

void VoxelMesh::swap() {
  osg::ref_ptr<osg::Vec3Array> vertexArrayTmp = m_vertexArrayTmp1;
  osg::ref_ptr<osg::Vec4Array> colorArrayTmp = m_colorArrayTmp1;
  osg::ref_ptr<osg::Vec3Array> normalsArrayTmp = m_normalsArrayTmp1;
  osg::ref_ptr<osg::Vec2Array> texCoordArrayTmp = m_texCoordArrayTmp1;

  m_vertexArrayTmp1 = m_vertexArrayTmp2;
  m_colorArrayTmp1 = m_colorArrayTmp2;
  m_normalsArrayTmp1 = m_normalsArrayTmp2;
  m_texCoordArrayTmp1 = m_texCoordArrayTmp2;

  m_vertexArrayTmp2 = vertexArrayTmp;
  m_colorArrayTmp2 = colorArrayTmp;
  m_normalsArrayTmp2 = normalsArrayTmp;
  m_texCoordArrayTmp2 = texCoordArrayTmp;
}

void VoxelMesh::build(VoxelMeshType::Enum type) {
  clear(m_vertexArray, m_colorArray, m_normalsArray, m_texCoordArray);
  clear(m_vertexArrayTmp1, m_colorArrayTmp1, m_normalsArrayTmp1,
        m_texCoordArrayTmp1);

  switch (type) {
    case VoxelMeshType::FULL:
      buildMeshFull(m_vertexArray, m_colorArray, m_normalsArray,
                    m_texCoordArray);
      break;
    case VoxelMeshType::REDUCED:
      buildMeshReduced(m_vertexArray, m_colorArray, m_normalsArray,
                       m_texCoordArray);
      break;
    default:
      return;
  }

  for (int i = 0; i < m_vertexArray->getNumElements(); i++) {
    m_vertexArrayTmp1->push_back(m_vertexArray->at(i));
    m_colorArrayTmp1->push_back(m_colorArray->at(i));
    m_normalsArrayTmp1->push_back(m_normalsArray->at(i));
    m_texCoordArrayTmp1->push_back(m_texCoordArray->at(i));
  }

  bind(m_vertexArrayTmp1, m_colorArrayTmp1, m_normalsArrayTmp1,
       m_texCoordArrayTmp1);
}

void VoxelMesh::crop(osg::Vec3i voxMin, osg::Vec3i voxMax) {
  clear(m_vertexArrayTmp2, m_colorArrayTmp2, m_normalsArrayTmp2,
        m_texCoordArrayTmp2);
  crop(m_vertexArray, m_colorArray, m_normalsArray, m_texCoordArray,
       m_vertexArrayTmp2, m_colorArrayTmp2, m_normalsArrayTmp2,
       m_texCoordArrayTmp2, voxMin, voxMax);
  bind(m_vertexArrayTmp2, m_colorArrayTmp2, m_normalsArrayTmp2,
       m_texCoordArrayTmp2);
  swap();
}

void VoxelMesh::clear(osg::ref_ptr<osg::Vec3Array> vertexArray,
                      osg::ref_ptr<osg::Vec4Array> colorArray,
                      osg::ref_ptr<osg::Vec3Array> normalsArray,
                      osg::ref_ptr<osg::Vec2Array> texCoordArray) {
  vertexArray->clear();
  colorArray->clear();
  normalsArray->clear();
  texCoordArray->clear();
  vertexArray->trim();
  colorArray->trim();
  normalsArray->trim();
  texCoordArray->trim();
}

void VoxelMesh::crop(osg::ref_ptr<osg::Vec3Array> srcVertices,
                     osg::ref_ptr<osg::Vec4Array> srcColors,
                     osg::ref_ptr<osg::Vec3Array> srcNormals,
                     osg::ref_ptr<osg::Vec2Array> srcTexCoords,
                     osg::ref_ptr<osg::Vec3Array> dstVertices,
                     osg::ref_ptr<osg::Vec4Array> dstColors,
                     osg::ref_ptr<osg::Vec3Array> dstNormals,
                     osg::ref_ptr<osg::Vec2Array> dstTexCoords,
                     osg::Vec3i voxMin, osg::Vec3i voxMax) {
  for (int i = 0; i < srcVertices->getNumElements(); i += 4) {
    osg::Vec3 v1 = srcVertices->at(i);
    osg::Vec3 v2 = srcVertices->at(i + 1);
    osg::Vec3 v3 = srcVertices->at(i + 2);
    osg::Vec3 v4 = srcVertices->at(i + 3);
    if (limitPolygon(&v1, &v2, &v3, &v4, voxMin,
                     voxMax + osg::Vec3i(1, 1, 1))) {
      dstVertices->push_back(v1);
      dstVertices->push_back(v2);
      dstVertices->push_back(v3);
      dstVertices->push_back(v4);
      dstColors->push_back(srcColors->at(i));
      dstColors->push_back(srcColors->at(i + 1));
      dstColors->push_back(srcColors->at(i + 2));
      dstColors->push_back(srcColors->at(i + 3));
      dstNormals->push_back(srcNormals->at(i));
      dstNormals->push_back(srcNormals->at(i + 1));
      dstNormals->push_back(srcNormals->at(i + 2));
      dstNormals->push_back(srcNormals->at(i + 3));
      // TODO(Texture coords not changed)
      dstTexCoords->push_back(srcTexCoords->at(i));
      dstTexCoords->push_back(srcTexCoords->at(i + 1));
      dstTexCoords->push_back(srcTexCoords->at(i + 2));
      dstTexCoords->push_back(srcTexCoords->at(i + 3));
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

void VoxelMesh::buildMeshReduced(osg::ref_ptr<osg::Vec3Array> vertices,
                                 osg::ref_ptr<osg::Vec4Array> colors,
                                 osg::ref_ptr<osg::Vec3Array> normals,
                                 osg::ref_ptr<osg::Vec2Array> texCoords) {
  int dims[3] = {m_voxels->getSizeX(), m_voxels->getSizeY(),
                 m_voxels->getSizeZ()};

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
                (0 <= x[d] ? m_voxels->getVoxelReadOnly(x[0], x[1], x[2]) : 0);
            voxel vb =
                (x[d] < dims[d] - 1 ? m_voxels->getVoxelReadOnly(
                                          x[0] + q[0], x[1] + q[1], x[2] + q[2])
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
                if (done) {
                  break;
                }
              }
              // Add quad
              x[u] = i;
              x[v] = j;
              int du[3] = {0, 0, 0};
              int dv[3] = {0, 0, 0};
              if (c > 0) {
                dv[v] = h;
                du[u] = w;

                osg::Vec3 v1(x[0], x[1], x[2]);
                osg::Vec3 v2(x[0] + du[0], x[1] + du[1], x[2] + du[2]);
                osg::Vec3 v3(x[0] + du[0] + dv[0], x[1] + du[1] + dv[1],
                             x[2] + du[2] + dv[2]);
                osg::Vec3 v4(x[0] + dv[0], x[1] + dv[1], x[2] + dv[2]);

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

                osg::Vec4 color = m_colorSet->getColor(c);

                vertices->push_back(v1);
                vertices->push_back(v2);
                vertices->push_back(v3);
                vertices->push_back(v4);

                colors->push_back(color);
                colors->push_back(color);
                colors->push_back(color);
                colors->push_back(color);

                normals->push_back(normal);
                normals->push_back(normal);
                normals->push_back(normal);
                normals->push_back(normal);

                texCoords->push_back(osg::Vec2(0, 0));
                texCoords->push_back(osg::Vec2(w, 0));
                texCoords->push_back(osg::Vec2(w, h));
                texCoords->push_back(osg::Vec2(0, h));

                // texCoords->push_back(osg::Vec2(0, 0));
                // texCoords->push_back(osg::Vec2(1, 0));
                // texCoords->push_back(osg::Vec2(1, 1));
                // texCoords->push_back(osg::Vec2(0, 1));
              }

              // Zero-out mask
              for (l = 0; l < h; ++l)
                for (k = 0; k < w; ++k) {
                  mask.at(n + k + l * dims[u]) = 0;
                }
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

void VoxelMesh::buildMeshFull(osg::ref_ptr<osg::Vec3Array> vertices,
                              osg::ref_ptr<osg::Vec4Array> colors,
                              osg::ref_ptr<osg::Vec3Array> normals,
                              osg::ref_ptr<osg::Vec2Array> texCoords) {
  for (int k = 0; k < static_cast<int>(m_voxels->getSizeZ()); ++k) {
    for (int j = 0; j < static_cast<int>(m_voxels->getSizeY()); ++j) {
      for (int i = 0; i < static_cast<int>(m_voxels->getSizeX()); ++i) {
        if (!m_voxels->isEmpty(i, j, k)) {
          voxel v = m_voxels->getVoxelReadOnly(i, j, k);
          osg::Vec4 color = m_colorSet->getColor(v);

          if (m_voxels->isEmpty(i + 1, j, k)) {
            vertices->push_back(osg::Vec3(i + 1, j, k));
            vertices->push_back(osg::Vec3(i + 1, j + 1, k));
            vertices->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            vertices->push_back(osg::Vec3(i + 1, j, k + 1));
            osg::Vec3 normal(1.0f, 0.0f, 0.0f);
            normals->push_back(normal);
            normals->push_back(normal);
            normals->push_back(normal);
            normals->push_back(normal);
            colors->push_back(color);
            colors->push_back(color);
            colors->push_back(color);
            colors->push_back(color);
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
          }
          if (m_voxels->isEmpty(i - 1, j, k)) {
            vertices->push_back(osg::Vec3(i, j, k));
            vertices->push_back(osg::Vec3(i, j + 1, k));
            vertices->push_back(osg::Vec3(i, j + 1, k + 1));
            vertices->push_back(osg::Vec3(i, j, k + 1));
            osg::Vec3 normal(-1.0f, 0.0f, 0.0f);
            normals->push_back(normal);
            normals->push_back(normal);
            normals->push_back(normal);
            normals->push_back(normal);
            colors->push_back(color);
            colors->push_back(color);
            colors->push_back(color);
            colors->push_back(color);
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
          }
          if (m_voxels->isEmpty(i, j + 1, k)) {
            vertices->push_back(osg::Vec3(i, j + 1, k));
            vertices->push_back(osg::Vec3(i + 1, j + 1, k));
            vertices->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            vertices->push_back(osg::Vec3(i, j + 1, k + 1));
            osg::Vec3 normal(0.0f, 1.0f, 0.0f);
            normals->push_back(normal);
            normals->push_back(normal);
            normals->push_back(normal);
            normals->push_back(normal);
            colors->push_back(color);
            colors->push_back(color);
            colors->push_back(color);
            colors->push_back(color);
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
          }
          if (m_voxels->isEmpty(i, j - 1, k)) {
            vertices->push_back(osg::Vec3(i, j, k));
            vertices->push_back(osg::Vec3(i + 1, j, k));
            vertices->push_back(osg::Vec3(i + 1, j, k + 1));
            vertices->push_back(osg::Vec3(i, j, k + 1));
            osg::Vec3 normal(0.0f, -1.0f, 0.0f);
            normals->push_back(normal);
            normals->push_back(normal);
            normals->push_back(normal);
            normals->push_back(normal);
            colors->push_back(color);
            colors->push_back(color);
            colors->push_back(color);
            colors->push_back(color);
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
          }
          if (m_voxels->isEmpty(i, j, k + 1)) {
            vertices->push_back(osg::Vec3(i, j, k + 1));
            vertices->push_back(osg::Vec3(i + 1, j, k + 1));
            vertices->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            vertices->push_back(osg::Vec3(i, j + 1, k + 1));
            osg::Vec3 normal(0.0f, 0.0f, 1.0f);
            normals->push_back(normal);
            normals->push_back(normal);
            normals->push_back(normal);
            normals->push_back(normal);
            colors->push_back(color);
            colors->push_back(color);
            colors->push_back(color);
            colors->push_back(color);
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
          }
          if (m_voxels->isEmpty(i, j, k - 1)) {
            vertices->push_back(osg::Vec3(i, j, k));
            vertices->push_back(osg::Vec3(i + 1, j, k));
            vertices->push_back(osg::Vec3(i + 1, j + 1, k));
            vertices->push_back(osg::Vec3(i, j + 1, k));
            osg::Vec3 normal(0.0f, 0.0f, -1.0f);
            normals->push_back(normal);
            normals->push_back(normal);
            normals->push_back(normal);
            normals->push_back(normal);
            colors->push_back(color);
            colors->push_back(color);
            colors->push_back(color);
            colors->push_back(color);
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
            texCoords->push_back(osg::Vec2(0, 0));
          }
        }
      }
    }
  }
}

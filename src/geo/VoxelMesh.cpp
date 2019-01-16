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
  setUseVertexBufferObjects(true);
  addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));
}

// Copy constructor
VoxelMesh::VoxelMesh(const VoxelMesh &voxmesh)
    : osg::Geometry(),
      m_transform(new osg::PositionAttitudeTransform()),
      m_vertexArray(voxmesh.m_vertexArray),
      m_colorArray(voxmesh.m_colorArray),
      m_normalsArray(voxmesh.m_normalsArray),
      m_polyMode(osg::PolygonMode::Mode::FILL) {
  m_voxels = new VoxelArray(*voxmesh.m_voxels);
  m_colorSet = new ColorSet(*voxmesh.m_colorSet);
  setUseVertexBufferObjects(true);
  addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));
}

// assignment operator
VoxelMesh &VoxelMesh::operator=(const VoxelMesh &voxmesh) {
  m_voxels = voxmesh.m_voxels;
  m_colorSet = voxmesh.m_colorSet;
  m_normalsArray = voxmesh.m_normalsArray;
  m_vertexArray = voxmesh.m_vertexArray;
  m_colorArray = voxmesh.m_colorArray;
  m_polyMode = voxmesh.m_polyMode;
  return *this;
}

void VoxelMesh::setPolygonMode(osg::PolygonMode::Mode mode) {
  m_polyMode = mode;
  osg::ref_ptr<osg::StateSet> stateset = getOrCreateStateSet();
  osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
  polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, mode);
  stateset->setAttributeAndModes(
      polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
}

void VoxelMesh::bindArrays() {
  m_vertexArray->dirty();
  m_colorArray->dirty();
  m_normalsArray->dirty();

  setVertexArray(m_vertexArray);
  setColorArray(m_colorArray);
  setNormalArray(m_normalsArray);
  setColorBinding(osg::Geometry::BIND_PER_VERTEX);
  setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

  osg::DrawArrays *drawArrays =
      static_cast<osg::DrawArrays *>(getPrimitiveSet(0));
  drawArrays->setCount(m_vertexArray->getNumElements());
  drawArrays->dirty();
  std::cout << m_vertexArray->getNumElements() << " vertices" << std::endl;

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
  // stateset->setMode(GL_COLOR_MATERIAL, osg::StateAttribute::ON);

  setPolygonMode(m_polyMode);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(this);

  if (m_transform->getNumChildren() > 0)
    m_transform->removeChildren(0, m_transform->getNumChildren());
  m_transform->addChild(geode);
}

void VoxelMesh::clearArrays() {
  m_vertexArray->clear();
  m_colorArray->clear();
  m_normalsArray->clear();

  m_vertexArray->trim();
  m_colorArray->trim();
  m_normalsArray->trim();
}

void VoxelMesh::buildMeshReduced(osg::Vec3i voxMin, osg::Vec3i voxMax) {
  clearArrays();

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

                m_vertexArray->push_back(v1);
                m_vertexArray->push_back(v2);
                m_vertexArray->push_back(v3);
                m_vertexArray->push_back(v4);

                osg::Vec4 color = m_colorSet->getColor(c);
                m_colorArray->push_back(color);
                m_colorArray->push_back(color);
                m_colorArray->push_back(color);
                m_colorArray->push_back(color);

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

                m_normalsArray->push_back(normal);
                m_normalsArray->push_back(normal);
                m_normalsArray->push_back(normal);
                m_normalsArray->push_back(normal);
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
  bindArrays();
}

void VoxelMesh::buildMeshFull(osg::Vec3i voxMin, osg::Vec3i voxMax) {
  clearArrays();

  std::cout << "min: " << voxMin.x() << ", " << voxMin.y() << ", " << voxMin.z()
            << " max: " << voxMax.x() << ", " << voxMax.y() << ", "
            << voxMax.z() << ", vox: " << m_voxels->getSizeX() << ", "
            << m_voxels->getSizeY() << ", " << m_voxels->getSizeZ()
            << std::endl;

  for (int k = 0; k < static_cast<int>(m_voxels->getSizeZ()); ++k) {
    for (int j = 0; j < static_cast<int>(m_voxels->getSizeY()); ++j) {
      for (int i = 0; i < static_cast<int>(m_voxels->getSizeX()); ++i) {
        if (i < voxMin.x() || (j < voxMin.y()) || (k < voxMin.z()) ||
            (i > voxMax.x()) || (j > voxMax.y()) || (k > voxMax.z()))
          continue;
        if (!m_voxels->isEmpty(i, j, k)) {
          voxel v = m_voxels->getVoxelReadOnly(i, j, k);
          osg::Vec4 color = m_colorSet->getColor(v);

          if (m_voxels->isEmpty(i + 1, j, k)) {
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k + 1));
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            osg::Vec3 normal(1.0f, 0.0f, 0.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
          if (m_voxels->isEmpty(i - 1, j, k)) {
            m_vertexArray->push_back(osg::Vec3(i, j, k));
            m_vertexArray->push_back(osg::Vec3(i, j + 1, k));
            m_vertexArray->push_back(osg::Vec3(i, j + 1, k + 1));
            m_vertexArray->push_back(osg::Vec3(i, j, k + 1));
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            osg::Vec3 normal(-1.0f, 0.0f, 0.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
          if (m_voxels->isEmpty(i, j + 1, k)) {
            m_vertexArray->push_back(osg::Vec3(i, j + 1, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            m_vertexArray->push_back(osg::Vec3(i, j + 1, k + 1));
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            osg::Vec3 normal(0.0f, 1.0f, 0.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
          if (m_voxels->isEmpty(i, j - 1, k)) {
            m_vertexArray->push_back(osg::Vec3(i, j, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k + 1));
            m_vertexArray->push_back(osg::Vec3(i, j, k + 1));
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            osg::Vec3 normal(0.0f, -1.0f, 0.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
          if (m_voxels->isEmpty(i, j, k + 1)) {
            m_vertexArray->push_back(osg::Vec3(i, j, k + 1));
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k + 1));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            m_vertexArray->push_back(osg::Vec3(i, j + 1, k + 1));
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            osg::Vec3 normal(0.0f, 0.0f, 1.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
          if (m_voxels->isEmpty(i, j, k - 1)) {
            m_vertexArray->push_back(osg::Vec3(i, j, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k));
            m_vertexArray->push_back(osg::Vec3(i, j + 1, k));
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            osg::Vec3 normal(0.0f, 0.0f, -1.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
        }
      }
    }
  }
  bindArrays();
}

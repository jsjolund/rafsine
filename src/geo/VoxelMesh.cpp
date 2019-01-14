#include "VoxelMesh.hpp"

// Constructor with an existing voxel array
VoxelMesh::VoxelMesh(VoxelArray *voxels)
    : osg::Geometry(),
      m_transform(new osg::PositionAttitudeTransform()),
      m_voxels(voxels) {
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
      m_normalsArray(voxmesh.m_normalsArray) {
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
  return *this;
}

/**
 * @brief
 * https://github.com/mikolalysenko/mikolalysenko.github.com/blob/master/MinecraftMeshes2/js/greedy.js
 *
 * @param voxMin
 * @param voxMax
 */
void VoxelMesh::buildMeshReduced(osg::Vec3i voxMin, osg::Vec3i voxMax) {
  m_vertexArray->clear();
  m_colorArray->clear();

  m_vertexArray->trim();
  m_colorArray->trim();

  int dims[3] = {m_voxels->getSizeX(), m_voxels->getSizeY(),
                 m_voxels->getSizeZ()};

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
          voxel a =
              (0 <= x[d] ? m_voxels->getVoxelReadOnly(x[0], x[1], x[2]) : 0);
          voxel b =
              (x[d] < dims[d] - 1 ? m_voxels->getVoxelReadOnly(
                                        x[0] + q[0], x[1] + q[1], x[2] + q[2])
                                  : 0);
          if (a == b) {
            mask.at(n) = 0;
          } else if (a) {
            mask.at(n) = a;
          } else {
            mask.at(n) = -b;
          }
        }
      // Increment x[d]
      ++x[d];
      // Generate mesh for mask using lexicographic ordering
      n = 0;
      for (j = 0; j < dims[v]; ++j)
        for (i = 0; i < dims[u];) {
          int c = mask.at(n);
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
            } else {
              c = -c;
              du[v] = h;
              dv[u] = w;
            }
            int vertex_count = m_vertexArray->getNumElements();
            m_vertexArray->push_back(osg::Vec3(x[0], x[1], x[2]));
            m_vertexArray->push_back(
                osg::Vec3(x[0] + du[0], x[1] + du[1], x[2] + du[2]));
            m_vertexArray->push_back(osg::Vec3(x[0] + du[0] + dv[0],
                                               x[1] + du[1] + dv[1],
                                               x[2] + du[2] + dv[2]));
            m_vertexArray->push_back(
                osg::Vec3(x[0] + dv[0], x[1] + dv[1], x[2] + dv[2]));
            osg::Vec4 color(0.5, 0.5, 0.5, 1);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            // faces.push([vertex_count, vertex_count+1, vertex_count+2, c]);
            // faces.push([vertex_count, vertex_count+2, vertex_count+3, c]);

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
  setVertexArray(m_vertexArray);
  setColorArray(m_colorArray);
  // setNormalArray(m_normalsArray);
  setColorBinding(osg::Geometry::BIND_PER_VERTEX);
  // setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

  osg::DrawArrays *drawArrays =
      static_cast<osg::DrawArrays *>(getPrimitiveSet(0));
  drawArrays->setCount(m_vertexArray->getNumElements());
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
  // stateset->setMode(GL_COLOR_MATERIAL, osg::StateAttribute::ON);

  // Filled ploygons
  osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
  polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
  stateset->setAttributeAndModes(
      polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(this);

  if (m_transform->getNumChildren() > 0)
    m_transform->removeChildren(0, m_transform->getNumChildren());
  m_transform->addChild(geode);
}

// build the mesh for the voxel array
void VoxelMesh::buildMesh(osg::Vec3i voxMin, osg::Vec3i voxMax) {
  m_vertexArray->clear();
  m_colorArray->clear();
  m_normalsArray->clear();

  m_vertexArray->trim();
  m_colorArray->trim();
  m_normalsArray->trim();

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

  setVertexArray(m_vertexArray);
  setColorArray(m_colorArray);
  setNormalArray(m_normalsArray);
  setColorBinding(osg::Geometry::BIND_PER_VERTEX);
  setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

  osg::DrawArrays *drawArrays =
      static_cast<osg::DrawArrays *>(getPrimitiveSet(0));
  drawArrays->setCount(m_vertexArray->getNumElements());
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
  // stateset->setMode(GL_COLOR_MATERIAL, osg::StateAttribute::ON);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(this);

  if (m_transform->getNumChildren() > 0)
    m_transform->removeChildren(0, m_transform->getNumChildren());
  m_transform->addChild(geode);
}

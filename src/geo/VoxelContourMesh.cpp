#include "VoxelContourMesh.hpp"

VoxelContourMesh::VoxelContourMesh(VoxelArray *voxels) : VoxelMesh(voxels)
{
  removePrimitiveSet(0, 1);
  addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, 0));
}

//build the mesh for the voxel array
void VoxelContourMesh::buildMesh()
{
  m_vertexArray->clear();
  m_vertexArray->trim();

  for (int k = 1; k < int(m_voxels->getSizeZ()) - 1; ++k)
    for (int j = 1; j < int(m_voxels->getSizeY()) - 1; ++j)
      for (int i = 1; i < int(m_voxels->getSizeX()) - 1; ++i)
      {
        if (!m_voxels->isEmpty(i, j, k))
        {
          voxel v = m_voxels->getVoxelReadOnly(i, j, k);
          col3 col_col3 = m_colorSet->getColor(v);
          osg::Vec4 col_vec3r(col_col3.r / 255., col_col3.g / 255., col_col3.b / 255., 1.0);
          //compute the number of empty neithbour
          int n = 0;
          if (m_voxels->isEmpty(i + 1, j, k))
            n++;
          if (m_voxels->isEmpty(i - 1, j, k))
            n++;
          if (m_voxels->isEmpty(i, j + 1, k))
            n++;
          if (m_voxels->isEmpty(i, j - 1, k))
            n++;
          if (m_voxels->isEmpty(i, j, k + 1))
            n++;
          if (m_voxels->isEmpty(i, j, k - 1))
            n++;
          if (n == 2)
            m_vertexArray->push_back(osg::Vec3(i, j, k));
        }
      }

  setVertexArray(m_vertexArray);

  osg::DrawArrays *drawArrays = static_cast<osg::DrawArrays *>(getPrimitiveSet(0));
  drawArrays->setCount(m_vertexArray->getNumElements());
  drawArrays->dirty();

  osg::ref_ptr<osg::StateSet> stateset = getOrCreateStateSet();
  stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(this);

  if (m_transform->getNumChildren() > 0)
    m_transform->removeChildren(0, m_transform->getNumChildren());
  m_transform->addChild(geode);
}
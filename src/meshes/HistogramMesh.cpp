#include "HistogramMesh.hpp"

void HistogramMesh::update(const thrust::host_vector<real_t>& histogram) {
  m_histogram.clear();
  m_histogram.resize(histogram.size());
  thrust::copy(histogram.begin(), histogram.end(), m_histogram.begin());

  int depth = 0;
  m_vertices->clear();
  m_vertices->resize(0);
  float w = m_width / histogram.size();
  for (size_t i = 0; i < histogram.size(); i++) {
    float h = m_height * histogram[i];
    float x = w * i;
    float y = 0;
    m_vertices->push_back(osg::Vec3(x, y, depth));
    m_vertices->push_back(osg::Vec3(x + w, y, depth));
    m_vertices->push_back(osg::Vec3(x + w, y + h, depth));
    m_vertices->push_back(osg::Vec3(x, y + h, depth));
  }
  m_vertices->dirty();

  setVertexArray(m_vertices);
  m_transform->setPivotPoint(osg::Vec3d(m_width / 2, 0, 0));
  m_transform->setAttitude(osg::Quat(0, osg::Vec3d(0, 0, 1)));
  m_transform->setPosition(osg::Vec3d(m_width / 2, SLICE_GRADIENT_HEIGHT, 0));

  osg::DrawArrays* drawArrays =
      static_cast<osg::DrawArrays*>(getPrimitiveSet(0));
  drawArrays->setCount(m_vertices->getNumElements());
  drawArrays->dirty();
}

HistogramMesh::HistogramMesh()
    : m_vertices(new osg::Vec3Array()), m_histogram(1) {
  m_histogram[0] = 0.0;

  setUseVertexBufferObjects(true);
  addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));

  osg::StateSet* stateset = getOrCreateStateSet();
  stateset->setAttributeAndModes(
      new osg::BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
  stateset->setRenderBinDetails(INT_MAX - 1, "RenderBin");

  osg::ref_ptr<osg::Material> mat = new osg::Material();
  mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
                   osg::Vec4f(1.0f, 1.0f, 1.0f, 0.7f));
  mat->setColorMode(osg::Material::ColorMode::AMBIENT_AND_DIFFUSE);

  stateset->setAttribute(mat.get(), osg::StateAttribute::Values::ON);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(this);
  m_transform = new osg::PositionAttitudeTransform();
  m_transform->addChild(geode);
}

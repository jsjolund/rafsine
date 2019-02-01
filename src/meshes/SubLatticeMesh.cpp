#include "SubLatticeMesh.hpp"

SubLatticeMesh::~SubLatticeMesh() { delete m_colorSet; }

void SubLatticeMesh::setProperties(osg::ref_ptr<osg::ShapeDrawable> drawable) {
  osg::ref_ptr<osg::StateSet> stateset = drawable->getOrCreateStateSet();
  // Filled ploygons
  osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
  polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);
  stateset->setAttributeAndModes(
      polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
  // Lighting
  stateset->setMode(GL_LIGHTING,
                    osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
  // Material
  osg::ref_ptr<osg::Material> material = new osg::Material;
  material->setAmbient(osg::Material::Face::FRONT_AND_BACK,
                       osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 1.0f);
  material->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
                       osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.5f);
  material->setEmission(osg::Material::Face::FRONT_AND_BACK,
                        osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.1f);
  material->setColorMode(osg::Material::ColorMode::AMBIENT_AND_DIFFUSE);
  stateset->setAttributeAndModes(
      material, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
  // Transparency
  stateset->setAttributeAndModes(
      new osg::BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
}

void SubLatticeMesh::addLabel(osg::Vec3d center, std::string content) {
  osg::ref_ptr<osg::PositionAttitudeTransform> transform =
      new osg::PositionAttitudeTransform();
  osg::ref_ptr<osgText::Text> text = new BillboardText();
  text->setBoundingBoxColor(osg::Vec4(0.0f, 0.0f, 0.0f, 0.5f));
  text->setCharacterSizeMode(osgText::Text::SCREEN_COORDS);
  text->setDrawMode(osgText::Text::TEXT | osgText::Text::ALIGNMENT |
                    osgText::Text::FILLEDBOUNDINGBOX);
  text->setAlignment(osgText::Text::LEFT_TOP);
  transform->addChild(text);
  transform->setPosition(center);
  addChild(transform);
  text->setText(content);
}

SubLatticeMesh::SubLatticeMesh(unsigned int latticeSizeX,
                               unsigned int latticeSizeY,
                               unsigned int latticeSizeZ,
                               unsigned int subLattices, float alpha)
    : DistributedLattice(latticeSizeX, latticeSizeY, latticeSizeZ, subLattices),
      osg::Geode(),
      m_colorSet(new ColorSet()) {
  const int numSubLattices = getNumSubLatticesTotal();
  for (int i = 0; i < numSubLattices; i++) {
    SubLattice subLattice = m_subLattices[i];

    glm::ivec3 min = subLattice.getLatticeMin();
    glm::ivec3 size = subLattice.getLatticeDims();
    glm::vec3 c =
        glm::vec3(min) + glm::vec3(size.x * 0.5f, size.y * 0.5f, size.z * 0.5f);

    // Create boxes
    {
      osg::ref_ptr<osg::ShapeDrawable> drawable = new osg::ShapeDrawable(
          new osg::Box(osg::Vec3d(c.x, c.y, c.z), size.x, size.y, size.z));
      osg::Vec4 color = m_colorSet->getColor(i + 2);
      color.a() *= alpha;
      drawable->setColor(color);
      addDrawable(drawable);
      // Set box visibility properties
      setProperties(drawable);
    }

    // Create labels
    std::stringstream ss;
    ss << "GPU" << getSubLatticeDevice(subLattice);
    addLabel(osg::Vec3d(c.x, c.y, c.z), ss.str());
  }
}

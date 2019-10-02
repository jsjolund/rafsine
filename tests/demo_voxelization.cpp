#include <osg/ArgumentParser>
#include <osg/Material>
#include <osg/PolygonMode>
#include <osg/Vec4>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "box_triangle/aabb_triangle_overlap.h"
#include "triangle_point/poitri.h"
#include "triangle_ray/raytri.h"

#include "ColorSet.hpp"
#include "StlFile.hpp"

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  std::string pathString;
  if (!args.read("-i", pathString)) {
    std::cout << "-i path/to/stl_directory" << std::endl;
    return -1;
  }

  osg::ref_ptr<osg::Group> root = new osg::Group;

  boost::filesystem::path input(pathString);
  boost::filesystem::directory_iterator end;
  ColorSet colorSet;
  int numModels = 0;

  for (boost::filesystem::directory_iterator it(input); it != end; ++it) {
    boost::filesystem::path filePath = it->path();
    if (filePath.extension().string() == ".stl") {
      numModels++;
      stl_file::StlFile solid(filePath.string());
      std::cout << solid.name << ": " << solid.vertices.size() << " vertices, "
                << solid.normals.size() << " normals" << std::endl;

      osg::Geode* geode = new osg::Geode();

      osg::Geometry* geom = new osg::Geometry();
      geom->setUseVertexBufferObjects(true);
      osg::Vec3Array* vertices = new osg::Vec3Array;
      for (int i = 0; i < solid.vertices.size(); i += 9) {
        float v0x = solid.vertices.at(i + 0);
        float v0y = solid.vertices.at(i + 1);
        float v0z = solid.vertices.at(i + 2);

        float v1x = solid.vertices.at(i + 3);
        float v1y = solid.vertices.at(i + 4);
        float v1z = solid.vertices.at(i + 5);

        float v2x = solid.vertices.at(i + 6);
        float v2y = solid.vertices.at(i + 7);
        float v2z = solid.vertices.at(i + 8);

        vertices->push_back(osg::Vec3(v0x, v0y, v0z));
        vertices->push_back(osg::Vec3(v1x, v1y, v1z));
        vertices->push_back(osg::Vec3(v2x, v2y, v2z));
      }

      osg::DrawArrays* drawArrays = new osg::DrawArrays(
          osg::PrimitiveSet::TRIANGLES, 0, vertices->size());
      geom->addPrimitiveSet(drawArrays);
      geom->setVertexArray(vertices);

      osg::Vec3Array* normals = new osg::Vec3Array;
      for (int i = 0; i < solid.normals.size(); i += 3) {
        float n0 = solid.normals.at(i);
        float n1 = solid.normals.at(i + 1);
        float n2 = solid.normals.at(i + 2);
        normals->push_back(osg::Vec3(n0, n1, n2));
        normals->push_back(osg::Vec3(n0, n1, n2));
        normals->push_back(osg::Vec3(n0, n1, n2));
      }
      geom->setNormalArray(normals, osg::Array::BIND_PER_VERTEX);

      osg::Vec4Array* colors = new osg::Vec4Array;
      colors->push_back(colorSet.getColor(numModels));
      geom->setColorArray(colors, osg::Array::BIND_OVERALL);

      geode->addDrawable(geom);
      osg::ref_ptr<osg::StateSet> stateset = geom->getOrCreateStateSet();

      stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
      stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);
      osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
      polymode->setMode(osg::PolygonMode::FRONT_AND_BACK,
                        osg::PolygonMode::FILL);
      stateset->setAttributeAndModes(polymode, osg::StateAttribute::ON);

      osg::ref_ptr<osg::Material> mat = new osg::Material();
      mat->setAmbient(osg::Material::Face::FRONT_AND_BACK,
                      osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 1.0f);
      mat->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
                      osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.5f);
      mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
                       osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.1f);
      mat->setColorMode(osg::Material::ColorMode::AMBIENT_AND_DIFFUSE);

      stateset->setAttribute(mat.get(), osg::StateAttribute::Values::ON);

      vertices->dirty();
      normals->dirty();
      drawArrays->dirty();

      root->addChild(geode);
    }
  }

  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  viewer.addEventHandler(new osgViewer::StatsHandler);
  return viewer.run();
}

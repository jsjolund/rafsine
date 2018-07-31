#pragma once

#include <osgViewer/Viewer>
#include <osgViewer/CompositeViewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/TrackballManipulator>
#include <osgDB/ReadFile>
#include <osg/PositionAttitudeTransform>

#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>

#include "SliceRender.hpp"
#include "PickHandler.hpp"
#include "../geo/VoxelMesh.hpp"

// Which quantity to display
namespace DisplayQuantity
{
enum Enum
{
  VELOCITY_NORM,
  DENSITY,
  TEMPERATURE
};
}

namespace DisplayMode
{
enum Enum
{
  SLICE,
  VOX_GEOMETRY
};
}
class AdapterWidget : public Fl_Gl_Window
{
public:
  AdapterWidget(int x, int y, int w, int h, const char *label) : Fl_Gl_Window(x, y, w, h, label)
  {
    _gw = new osgViewer::GraphicsWindowEmbedded(x, y, w, h);
  }
  virtual ~AdapterWidget() {}

  inline osgViewer::GraphicsWindow *getGraphicsWindow() { return _gw.get(); }
  inline const osgViewer::GraphicsWindow *getGraphicsWindow() const { return _gw.get(); }

  void resize(int x, int y, int w, int h) override;

protected:
  osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> _gw;
};

class GLWindow : public osgViewer::Viewer, public AdapterWidget
{
private:
  VoxelMesh *voxmesh_;
  osg::Geometry *voxGeo;
  osg::PositionAttitudeTransform *voxGeoTransform;
  DisplayMode::Enum displayMode_;
  DisplayQuantity::Enum displayQuantity_;
  vec3<int> vox_size_, vox_max_, vox_min_, slice_pos_;

  cudaStream_t renderStream_;
  SliceRender *sliceX_, *sliceY_, *sliceZ_, *sliceC_;

public:
  void setCudaRenderStream(cudaStream_t stream) { renderStream_ = stream; };
  void redrawVoxelMesh();
  void setVoxelMesh(VoxelMesh *mesh);
  void sliceXup();
  void sliceXdown();
  void sliceYup();
  void sliceYdown();
  void sliceZup();
  void sliceZdown();
  void setSliceXpos(int pos);
  void setSliceYpos(int pos);
  void setSliceZpos(int pos);

  GLWindow(int x, int y, int w, int h, const char *label = 0);

protected:
  int handle(int event) override;
  void draw();
  
};

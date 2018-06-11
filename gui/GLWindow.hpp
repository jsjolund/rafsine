#pragma once

#include <osgViewer/Viewer>
#include <osgViewer/CompositeViewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/TrackballManipulator>
#include <osgDB/ReadFile>

#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>

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

    void resize(int x, int y, int w, int h);

  protected:
    int handle(int event);
    osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> _gw;
};

class GLWindow : public osgViewer::Viewer, public AdapterWidget
{
  public:
    GLWindow(int x, int y, int w, int h, const char *label = 0);

  protected:
    void draw() { frame(); }
};

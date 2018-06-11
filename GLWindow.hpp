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
    GLWindow(int x, int y, int w, int h, const char *label = "test") : AdapterWidget(x, y, w, h, label)
    {
        getCamera()->setViewport(new osg::Viewport(0, 0, w, h));
        getCamera()->setProjectionMatrixAsPerspective(30.0f, static_cast<double>(w) / static_cast<double>(h), 1.0, 10000.0);
        getCamera()->setGraphicsContext(getGraphicsWindow());
        getCamera()->setDrawBuffer(GL_BACK);
        getCamera()->setReadBuffer(GL_BACK);
        setThreadingModel(osgViewer::Viewer::SingleThreaded);
    }

  protected:
    void draw() { frame(); }
};

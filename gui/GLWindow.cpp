#include "GLWindow.hpp"

void AdapterWidget::resize(int x, int y, int w, int h)
{
    _gw->getEventQueue()->windowResize(x, y, w, h);
    _gw->resized(x, y, w, h);

    Fl_Gl_Window::resize(x, y, w, h);
}

int AdapterWidget::handle(int event)
{
    switch (event)
    {
    case FL_FOCUS:
        std::cout << "focus" << std::endl;
        return 1;
    case FL_PUSH:
        Fl::focus(this);
        std::cout << "push" << std::endl;
        _gw->getEventQueue()->mouseButtonPress(Fl::event_x(), Fl::event_y(), Fl::event_button());
        return 1;
    case FL_MOVE:
        return 1;
    case FL_DRAG:
        std::cout << "drag" << std::endl;
        _gw->getEventQueue()->mouseMotion(Fl::event_x(), Fl::event_y());
        return 1;
    case FL_RELEASE:
        std::cout << "release" << std::endl;
        _gw->getEventQueue()->mouseButtonRelease(Fl::event_x(), Fl::event_y(), Fl::event_button());
        return 1;
    case FL_KEYDOWN:
        std::cout << "keydown" << std::endl;
        _gw->getEventQueue()->keyPress((osgGA::GUIEventAdapter::KeySymbol)Fl::event_key());
        return 1;
    case FL_KEYUP:
        std::cout << "keyup" << std::endl;
        _gw->getEventQueue()->keyRelease((osgGA::GUIEventAdapter::KeySymbol)Fl::event_key());
        return 1;
    default:
        // pass other events to the base class
        return Fl_Gl_Window::handle(event);
    }
}

GLWindow::GLWindow(int x, int y, int w, int h, const char *label)
    : AdapterWidget(x, y, w, h, label)
{
    getCamera()->setViewport(new osg::Viewport(0, 0, w, h));
    getCamera()->setProjectionMatrixAsPerspective(30.0f, static_cast<double>(w) / static_cast<double>(h), 1.0, 10000.0);
    getCamera()->setGraphicsContext(getGraphicsWindow());
    getCamera()->setDrawBuffer(GL_BACK);
    getCamera()->setReadBuffer(GL_BACK);
    setThreadingModel(osgViewer::Viewer::SingleThreaded);
    setCameraManipulator(new osgGA::TrackballManipulator);
    addEventHandler(new osgViewer::StatsHandler);
    
    addEventHandler(new PickHandler());
}
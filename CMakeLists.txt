cmake_minimum_required(VERSION 3.10)
project(rafsine LANGUAGES CXX CUDA)

# Set which geometry to generate
# set(problem pod2)

# Workspace settings
add_definitions(-DWITH_OPENGL -DVERBOSE)
# set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CURRENT_BINARY_DIR "${PROJECT_SOURCE_DIR}/cmake-build-debug")
set(OpenGL_GL_PREFERENCE "GLVND")

# These are only needed to make the cmake plugin in vscode work with intellisense
include_directories("/home/user/code/rafsine/src")
include_directories("/home/ubuntu/rafsine/src")
include_directories("/usr/include/.")
include_directories("/usr/include/c++/6")
include_directories("/usr/include/x86_64-linux-gnu")
include_directories("/usr/include/x86_64-linux-gnu/c++/6")
include_directories("/usr/lib/gcc/x86_64-linux-gnu/6/include")

# C/C++ Compiler settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-6")
set(CMAKE_CXX_FLAGS "-fpermissive ")
# set(CMAKE_CXX_FLAGS "-fpermissive -Wall -Wextra")

# CUDA Compiler settings
set(CUDACXX=/usr/local/cuda/bin/nvcc)
set(CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61") # GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# set(CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52") # Quadro M6000 , GeForce 900, GTX-970, GTX-980, GTX Titan X
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -shared -Xcompiler -fPIC")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ccbin gcc-6 -std=c++11")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g --ptxas-options=-v -lineinfo")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

# Ignore glm warnings about "annotation on a defaulted function"
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

set(SOURCE_FILES
        main.cpp
        ResizeBar.cpp
        GLWindow.cpp
        MainMenu.cpp
        )

set(HEADER_FILES
        main.hpp
        ResizeBar.hpp
        GLWindow.hpp
        DataTable.hpp
        TreeView.hpp
        MainMenu.hpp
        )

message(STATUS "CMAKE_INCLUDE_PATH: " ${CMAKE_INCLUDE_PATH})
message(STATUS "CMAKE_LIBRARY_PATH: " ${CMAKE_LIBRARY_PATH})
message(STATUS "CMAKE_SYSTEM: " ${CMAKE_SYSTEM})
message(STATUS "CMAKE_SYSTEM_PROCESSOR: " ${CMAKE_SYSTEM_PROCESSOR})
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR})

# Search for system libraries

find_package(OpenSceneGraph REQUIRED osgDB osgUtil osgViewer osgText osgGA osgTerrain osgManipulator)
include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})
message(STATUS "OPENSCENEGRAPH_INCLUDE_DIRS: ${OPENSCENEGRAPH_INCLUDE_DIRS}")

find_package(FLTK REQUIRED)
include_directories(${FLTK_INCLUDE_DIRS})
message(STATUS "FLTK_INCLUDE_DIRS: ${FLTK_INCLUDE_DIRS}")
if (NOT FLTK_FOUND)
    message(FATAL_ERROR "FLTK not found")
endif ()

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS system filesystem iostreams REQUIRED)
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
message(STATUS "Boost_VERSION: ${Boost_VERSION}")
include_directories(${Boost_INCLUDE_DIRS})
if (NOT Boost_FOUND)
    message(FATAL_ERROR "Boost not found")
endif ()

find_package(CUDA REQUIRED)
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
include_directories(${CUDA_INCLUDE_DIRS})
if (NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found")
endif ()

find_package(Lua "5.1" EXACT REQUIRED)
message(STATUS "LUA_INCLUDE_DIR: ${LUA_INCLUDE_DIR}")
message(STATUS "LUA_LIBRARIES: ${LUA_LIBRARIES}")
message(STATUS "LUA_VERSION: ${LUA_VERSION_STRING}")
include_directories(${LUA_INCLUDE_DIRS})
if (NOT LUA_FOUND)
    message(FATAL_ERROR "Lua not found")
endif ()

find_package(PkgConfig REQUIRED)
if (PKG_CONFIG_FOUND)
    message(STATUS "PkgConfig found")
else ()
    message(FATAL_ERROR "PkgConfig not found")
endif ()

find_package(GLEW REQUIRED)
message(STATUS "GLEW_INCLUDE_DIR: ${GLEW_INCLUDE_DIR}")
message(STATUS "GLEW_LIBRARIES: ${GLEW_LIBRARIES}")
include_directories(${GLEW_INCLUDE_DIR})
if (NOT GLEW_FOUND)
    message(FATAL_ERROR "GLEW not found")
endif ()

find_package(OpenGL REQUIRED)
message(STATUS "OpenGL_OpenGL_FOUND: ${OpenGL_OpenGL_FOUND}")
message(STATUS "OPENGL_glu_LIBRARY: ${OPENGL_glu_LIBRARY}")
message(STATUS "OPENGL_glx_LIBRARY: ${OPENGL_glx_LIBRARY}")
message(STATUS "OPENGL_opengl_LIBRARY: ${OPENGL_opengl_LIBRARY}")
message(STATUS "OPENGL_INCLUDE_DIR: ${OPENGL_INCLUDE_DIR}")
message(STATUS "OPENGL_LIBRARIES: ${OPENGL_LIBRARIES}")
include_directories(${OPENGL_INCLUDE_DIR})
if (NOT OPENGL_FOUND)
    message(FATAL_ERROR "OpenGL not found")
endif ()

find_package(JPEG REQUIRED)
message(STATUS "JPEG_INCLUDE_DIR: ${JPEG_INCLUDE_DIR}")
message(STATUS "JPEG_LIBRARIES: ${JPEG_LIBRARIES}")
include_directories(${JPEG_INCLUDE_DIR})
if (NOT JPEG_FOUND)
    message(FATAL_ERROR "JPEG not found")
endif ()

find_package(X11 REQUIRED)
message(STATUS "X11_INCLUDE_DIR: ${X11_INCLUDE_DIR}")
message(STATUS "X11_LIBRARIES: ${X11_LIBRARIES}")
include_directories(${X11_INCLUDE_DIR})
if (NOT X11_FOUND)
    message(FATAL_ERROR "X11 not found")
endif ()

find_package(glfw3 REQUIRED)
message(STATUS "GLFW3_LIBRARY: ${GLFW3_LIBRARY}")
if (NOT glfw3_FOUND)
    message(FATAL_ERROR "glfw3 not found")
endif ()

# find_package(glm REQUIRED)

find_package(GLUT REQUIRED)
message(STATUS "GLUT_INCLUDE_DIR: ${GLUT_INCLUDE_DIR}")
message(STATUS "GLUT_LIBRARIES: ${GLUT_LIBRARIES}")
include_directories(${GLUT_INCLUDE_DIR})
if (NOT GLUT_FOUND)
    message(FATAL_ERROR "GLUT not found")
endif ()

find_package(Threads REQUIRED)
message(STATUS "CMAKE_THREAD_LIBS_INIT: ${CMAKE_THREAD_LIBS_INIT}")
message(STATUS "CMAKE_USE_SPROC_INIT: ${CMAKE_USE_SPROC_INIT}")
message(STATUS "CMAKE_USE_WIN32_THREADS_INIT: ${CMAKE_USE_WIN32_THREADS_INIT}")
message(STATUS "CMAKE_USE_PTHREADS_INIT: ${CMAKE_USE_PTHREADS_INIT}")
message(STATUS "CMAKE_HP_PTHREADS_INIT: ${CMAKE_HP_PTHREADS_INIT}")
if (NOT CMAKE_THREAD_LIBS_INIT)
    message(FATAL_ERROR "Pthread not found")
endif ()
add_definitions(-DHAVE_PTHREAD_H)

# Link the external libraries to the project
link_libraries(
        # "/usr/lib/liblua5.1.so;/usr/lib/libm.so"
        ${LUA_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${Boost_LIBRARIES}
        ${JPEG_LIBRARIES}
        ${X11_LIBRARIES}
        ${GLUT_Xi_LIBRARY}
        ${FLTK_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${OPENSCENEGRAPH_LIBRARIES}
)

add_library(rafsine STATIC ${HEADER_FILES})

# Request that rafsine be built with -std=c++11
# As this is a public compile feature anything that links to rafsine
# will also build with -std=c++11
target_compile_features(rafsine PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the rafsine
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(rafsine PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CXX)

# Print out some settings
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CUDA_PROPAGATE_HOST_FLAGS: ${CUDA_PROPAGATE_HOST_FLAGS}")
message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

# Add the executable to be compiled
set_source_files_properties(${SOURCE_FILES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
cuda_add_executable(rafsine-debug ${SOURCE_FILES})
target_link_libraries(rafsine-debug -lutil -lglut -lcuda -llua5.1 -lm -lc)
add_dependencies(rafsine-debug rafsine)

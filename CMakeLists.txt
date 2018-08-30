# --------------------------------------------------------------------
# |  Projects Settings                                               |
# --------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)
project(rafsine LANGUAGES CXX CUDA)
# set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CURRENT_BINARY_DIR "${PROJECT_SOURCE_DIR}/cmake-build-debug")
set(FIND_LIBRARY_USE_LIB64_PATHS on)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/ext/luawrapper")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")

# --------------------------------------------------------------------
# |  C/C++ Compiler settings                                         |
# -------------------------------------------------------------------- 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wno-attributes -fPIC")
set(CUDACXX="/usr/local/cuda/bin/nvcc –default-stream per-thread")
set(CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61")

# --------------------------------------------------------------------
# |  CUDA                                                            |
# -------------------------------------------------------------------- 
find_package(CUDA REQUIRED)
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
include_directories(${CUDA_INCLUDE_DIRS})
if (NOT CUDA_FOUND)
message(FATAL_ERROR "CUDA not found")
endif ()
add_definitions(-DHAVE_CUDA)

# --------------------------------------------------------------------
# |  OpenSceneGraph                                                  |
# -------------------------------------------------------------------- 
find_package(OSG REQUIRED osgDB osgUtil osgViewer osgText osgGA osgTerrain osgManipulator OpenThreads)
include_directories(${OSG_INCLUDE_DIR})
message(STATUS "OSG_INCLUDE_DIR: ${OSG_INCLUDE_DIR}")
message(STATUS "OSG_LIBRARY: ${OSG_LIBRARY}")
message(STATUS "OPENTHREADS_LIBRARY: ${OPENTHREADS_LIBRARY}")

# --------------------------------------------------------------------
# |  QT                                                              |
# -------------------------------------------------------------------- 
find_package(Qt5Widgets CONFIG REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# --------------------------------------------------------------------
# |  Boost                                                           |
# -------------------------------------------------------------------- 
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

# --------------------------------------------------------------------
# |  Lua                                                             |
# -------------------------------------------------------------------- 
find_package(Lua51)
# find_package(Lua52)
find_package(LuaJIT51)
message(STATUS "LUA_INCLUDE_DIR: ${LUA_INCLUDE_DIR}")
message(STATUS "LUA_LIBRARIES: ${LUA_LIBRARIES}")
include_directories(${LUA_INCLUDE_DIR})
if (NOT LUAJIT51_FOUND)
    message(FATAL_ERROR "Lua not found")
endif ()

# --------------------------------------------------------------------
# |  OpenGL                                                          |
# -------------------------------------------------------------------- 
set(OpenGL_GL_PREFERENCE "GLVND")
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

# --------------------------------------------------------------------
# |  Source files                                                    |
# -------------------------------------------------------------------- 
include_directories(include)
include_directories(ext/ordered-map)
include_directories(ext/luawrapper/include)

file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
file(GLOB_RECURSE HEADER_FILES include/*.hpp)

qt5_wrap_cpp(MOC_FILES ${HEADER_FILES})

set_source_files_properties(main.cpp ${SOURCE_FILES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
cuda_add_executable(rafsine main.cpp ${SOURCE_FILES} ${MOC_FILES})

qt5_use_modules(rafsine Core Widgets Gui OpenGL)

target_link_libraries(rafsine
    ${LUA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OSG_LIBRARY}
    ${OSGGA_LIBRARY}
    ${OSGVIEWER_LIBRARY}
    ${OPENTHREADS_LIBRARY}
    Qt5::Core
    Qt5::Gui
    Qt5::OpenGL
    Qt5::Widgets
    -lcuda
)

# Print out some settings
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CUDA_PROPAGATE_HOST_FLAGS: ${CUDA_PROPAGATE_HOST_FLAGS}")
message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
message(STATUS "CMAKE_INCLUDE_PATH: ${CMAKE_INCLUDE_PATH}")
message(STATUS "CMAKE_LIBRARY_PATH: ${CMAKE_LIBRARY_PATH}")
message(STATUS "CMAKE_SYSTEM: ${CMAKE_SYSTEM}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

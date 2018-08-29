cmake_minimum_required(VERSION 3.10)

project(rafsine LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CURRENT_BINARY_DIR "${PROJECT_SOURCE_DIR}/cmake-build-debug")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(FIND_LIBRARY_USE_LIB64_PATHS on)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/ext/luawrapper")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")

# C/C++ Compiler settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wno-attributes -fPIC")
set(CUDACXX="/usr/local/cuda/bin/nvcc –default-stream per-thread")
set(CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61") # GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4

include_directories("/usr/include/.")
include_directories("/usr/include/lua5.1")
include_directories("/usr/local/cuda-9.1/include")
include_directories("/usr/include/c++/6")
include_directories("/usr/include/x86_64-linux-gnu")
include_directories("/usr/include/x86_64-linux-gnu/c++/6")
include_directories("/usr/lib/gcc/x86_64-linux-gnu/6/include")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/ext/ordered-map)
include_directories(${PROJECT_SOURCE_DIR}/ext/luawrapper/include)

# Search for system libraries
find_package(CUDA REQUIRED)
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
include_directories(${CUDA_INCLUDE_DIRS})
if (NOT CUDA_FOUND)
message(FATAL_ERROR "CUDA not found")
endif ()
add_definitions(-DHAVE_CUDA)

find_package(OSG REQUIRED osgDB osgUtil osgViewer osgText osgGA osgTerrain osgManipulator OpenThreads)
include_directories(${OSG_INCLUDE_DIR})
message(STATUS "OSG_INCLUDE_DIR: ${OSG_INCLUDE_DIR}")
message(STATUS "OSG_LIBRARY: ${OSG_LIBRARY}")
message(STATUS "OPENTHREADS_LIBRARY: ${OPENTHREADS_LIBRARY}")

find_package(Qt5 REQUIRED COMPONENTS Core Gui OpenGL Widgets)
find_package(Qt5Widgets CONFIG REQUIRED)

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

find_package(Lua "5.1" EXACT REQUIRED)
message(STATUS "LUA_INCLUDE_DIR: ${LUA_INCLUDE_DIR}")
message(STATUS "LUA_LIBRARIES: ${LUA_LIBRARIES}")
message(STATUS "LUA_VERSION: ${LUA_VERSION_STRING}")
include_directories(${LUA_INCLUDE_DIRS})
if (NOT LUA_FOUND)
    message(FATAL_ERROR "Lua not found")
endif ()

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

set(SOURCE_FILES
        main.cpp
        src/cuda/CudaTexturedQuadGeometry.cpp
        src/cuda/CudaTextureSubloadCallback.cpp
        src/cuda/CudaGraphicsResource.cpp
        src/cuda/CudaTexture2D.cpp
        src/gui/CFDWidget.cpp
        src/gui/QtOSGWidget.cpp
        src/gui/SliceRender.cpp
        src/gui/PickHandler.cpp
        src/gui/CFDScene.cpp
        src/gui/MainWindow.cpp
        src/geo/VoxelGeometry.cpp
        src/geo/Voxel.cpp
        src/geo/VoxelMesh.cpp
        src/geo/VoxelDotMesh.cpp
        src/sim/DomainData.cpp
        src/sim/Kernel.cpp
        src/sim/KernelData.cpp
        src/sim/SimulationThread.cpp
        src/sim/BoundaryCondition.cpp
        src/sim/SimulationTimer.cpp
        )
        
set(HEADER_FILES
        include/CudaTexturedQuadGeometry.hpp
        include/CudaTextureSubloadCallback.hpp
        include/CudaGraphicsResource.hpp
        include/CudaTexture2D.hpp
        include/CudaUtils.hpp
        include/CFDWidget.hpp
        include/QtOSGWidget.hpp
        include/SliceRender.hpp
        include/PickHandler.hpp
        include/CFDScene.hpp
        include/VoxelGeometry.hpp
        include/Voxel.hpp
        include/VoxelMesh.hpp
        include/VoxelDotMesh.hpp
        include/Primitives.hpp
        include/ColorSet.hpp
        include/UnitConverter.hpp
        include/BoundaryCondition.hpp
        include/DomainData.hpp
        include/ErrorFormat.hpp
        include/DF_array_GPU.hpp
        include/CudaMathHelper.h
        include/Kernel.hpp
        include/KernelData.hpp
        include/SimulationThread.hpp
        include/SimulationTimer.hpp
        include/MainWindow.hpp
        )

# Link the external libraries to the project
link_libraries(
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
)
qt5_wrap_cpp(${SOURCE_FILES} ${HEADER_FILES})

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

# Add the executable to be compiled
set_source_files_properties(${SOURCE_FILES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
cuda_add_executable(rafsine-debug ${SOURCE_FILES} application.qrc)
target_link_libraries(rafsine-debug -lcuda)
add_dependencies(rafsine-debug rafsine)

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

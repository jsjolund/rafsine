set(CUDA_BIN_PATH "/usr/local/cuda-10.0")

# --------------------------------------------------------------------
# |  Projects Settings                                               |
# --------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)
project(rafsine LANGUAGES CXX CUDA)

enable_testing()
set(FIND_LIBRARY_USE_LIB64_PATHS on)
set(CMAKE_CURRENT_BINARY_DIR "${PROJECT_SOURCE_DIR}/cmake-build-debug")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/ext/luawrapper")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
# set(CMAKE_VERBOSE_MAKEFILE on)

# --------------------------------------------------------------------
# |  C/C++ Compiler settings                                         |
# --------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -fPIC -fopenmp -std=c++11")
set(CUDACXX="/usr/local/cuda/bin/nvcc")
set(CUDA_NVCC_FLAGS "-Xcompiler \
                -std=c++11 \
                --expt-relaxed-constexpr \
                -gencode arch=compute_61,code=sm_61 \
                -Xcudafe \"--display_error_number\" \
                -Xcudafe \"--diag_suppress=1427\" \
                -Xcudafe \"--diag_suppress=2906\"")

# --------------------------------------------------------------------
# |  Google Test                                                     |
# --------------------------------------------------------------------
add_subdirectory(${PROJECT_SOURCE_DIR}/ext/googletest)
include_directories(BEFORE ${PROJECT_SOURCE_DIR}/ext/googletest/googletest/include)
include(CTest) 

# --------------------------------------------------------------------
# |  CUDA                                                            |
# --------------------------------------------------------------------
find_package(CUDA 9.1 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
add_definitions(-DHAVE_CUDA)

# --------------------------------------------------------------------
# |  OpenSceneGraph                                                  |
# --------------------------------------------------------------------
find_package(OSG
             REQUIRED
             osgDB
             osgUtil
             osgViewer
             osgText
             osgGA
             osgManipulator)
include_directories(${OSG_INCLUDE_DIR})

# --------------------------------------------------------------------
# |  QT                                                              |
# --------------------------------------------------------------------
find_package(Qt5Widgets CONFIG REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY AUTOGEN_TARGETS_FOLDER AutoMoc)
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
find_package(Boost
             COMPONENTS system
                        filesystem
                        iostreams
                        REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# --------------------------------------------------------------------
# |  Lua                                                             |
# --------------------------------------------------------------------
find_package(Lua51)
find_package(Lua52)
find_package(LuaJIT51 REQUIRED)
include_directories(${LUA_INCLUDE_DIR})

# --------------------------------------------------------------------
# |  OpenGL                                                          |
# --------------------------------------------------------------------
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
# add_definitions(-DGLM_ENABLE_EXPERIMENTAL)

# --------------------------------------------------------------------
# |  Source files                                                    |
# --------------------------------------------------------------------
include_directories(include)
include_directories(ext/luawrapper/include)
include_directories(ext/googletest/googletest/include)

file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
file(GLOB_RECURSE HEADER_FILES include/*.hpp)

qt5_add_resources(RC_SRC "${PROJECT_SOURCE_DIR}/res.qrc")
qt5_wrap_cpp(MOC_HEADERS ${HEADER_FILES})

set_source_files_properties(main.cpp
                            ${SOURCE_FILES}
                            ${MOC_HEADERS}
                            PROPERTIES
                            CUDA_SOURCE_PROPERTY_FORMAT
                            OBJ)

cuda_add_library(rafsine_lib STATIC ${SOURCE_FILES} ${MOC_HEADERS})
target_link_libraries(rafsine_lib
                      ${LUA_LIBRARIES}
                      ${OPENGL_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${OSG_LIBRARY}
                      ${OSGTEXT_LIBRARY}
                      ${OSGDB_LIBRARY}
                      ${OSGGA_LIBRARY}
                      ${OSGVIEWER_LIBRARY}
                      ${OPENTHREADS_LIBRARY}
                      Qt5::Core
                      Qt5::Gui
                      Qt5::OpenGL
                      Qt5::Widgets
                      -lcuda)

# --------------------------------------------------------------------
# |  Main executable                                                 |
# --------------------------------------------------------------------
cuda_add_executable(rafsine main.cpp ${SOURCE_FILES} ${MOC_HEADERS} ${RC_SRC})
target_link_libraries(rafsine rafsine_lib)
qt5_use_modules(rafsine Core Widgets Gui OpenGL)

# --------------------------------------------------------------------
# |  Unit tests                                                      |
# --------------------------------------------------------------------
file(GLOB_RECURSE UNIT_TEST_FILES tests/test_*.cpp)

set_source_files_properties(${UNIT_TEST_FILES}
                            PROPERTIES
                            CUDA_SOURCE_PROPERTY_FORMAT
                            OBJ)

cuda_add_executable(unit-tests ${UNIT_TEST_FILES} ${SOURCE_FILES} ${MOC_HEADERS})

target_link_libraries(unit-tests rafsine_lib gtest gtest_main)

add_test(tests unit-tests)

# --------------------------------------------------------------------
# |  Other executables                                               |
# --------------------------------------------------------------------
set(TOPOLOGY_DEMO tests/demo_topology.cpp)

set_source_files_properties(${TOPOLOGY_DEMO}
                            PROPERTIES
                            CUDA_SOURCE_PROPERTY_FORMAT
                            OBJ)

cuda_add_executable(topology-demo
                    ${TOPOLOGY_DEMO}
                    ${SOURCE_FILES}
                    ${MOC_HEADERS})

target_link_libraries(topology-demo rafsine_lib)

# --------------------------------------------------------------------
# |  Post build                                                      |
# --------------------------------------------------------------------
add_custom_command(TARGET rafsine POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rafsine>
                           ${PROJECT_SOURCE_DIR})

# --------------------------------------------------------------------
# |  Debug print                                                     |
# --------------------------------------------------------------------
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CUDA_PROPAGATE_HOST_FLAGS: ${CUDA_PROPAGATE_HOST_FLAGS}")
message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
message(STATUS "CMAKE_SYSTEM: ${CMAKE_SYSTEM}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")

message(STATUS "OSG_INCLUDE_DIR: ${OSG_INCLUDE_DIR}")
message(STATUS "OSG_LIBRARY: ${OSG_LIBRARY}")
message(STATUS "OPENTHREADS_LIBRARY: ${OPENTHREADS_LIBRARY}")

message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
message(STATUS "Boost_VERSION: ${Boost_VERSION}")

message(STATUS "LUA_INCLUDE_DIR: ${LUA_INCLUDE_DIR}")
message(STATUS "LUA_LIBRARIES: ${LUA_LIBRARIES}")

message(STATUS "OpenGL_OpenGL_FOUND: ${OpenGL_OpenGL_FOUND}")
message(STATUS "OPENGL_glu_LIBRARY: ${OPENGL_glu_LIBRARY}")
message(STATUS "OPENGL_glx_LIBRARY: ${OPENGL_glx_LIBRARY}")
message(STATUS "OPENGL_opengl_LIBRARY: ${OPENGL_opengl_LIBRARY}")
message(STATUS "OPENGL_INCLUDE_DIR: ${OPENGL_INCLUDE_DIR}")
message(STATUS "OPENGL_LIBRARIES: ${OPENGL_LIBRARIES}")
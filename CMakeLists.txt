cmake_minimum_required(VERSION 3.10)

set(DOXYGEN_ENABLED off)

# --------------------------------------------------------------------
# |  C++/NVCC Compiler settings                                      |
# --------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER g++-6)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fopenmp -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-dev")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format")

set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \
                -Xcompiler \
                -std=c++11 \
                -lineinfo \
                -Xcudafe \"--display_error_number\" \
                -Xcudafe \"--diag_suppress=1427\" \
                -Xcudafe \"--diag_suppress=2886\" \
                -Xcudafe \"--diag_suppress=2906\" \
                -gencode arch=compute_61,code=sm_61")

# --------------------------------------------------------------------
# |  Projects Settings                                               |
# --------------------------------------------------------------------
project(rafsine LANGUAGES CXX CUDA)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake-modules")
set(FIND_LIBRARY_USE_LIB64_PATHS on)
enable_testing()

# --------------------------------------------------------------------
# |  CUDA                                                            |
# --------------------------------------------------------------------
find_package(CUDA 10.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
add_definitions(-DHAVE_CUDA)

# --------------------------------------------------------------------
# |  OpenSceneGraph                                                  |
# --------------------------------------------------------------------
find_package(OSG REQUIRED osgDB osgUtil osgViewer osgText osgGA osgManipulator)
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
# |  OpenGL                                                          |
# --------------------------------------------------------------------
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
# add_definitions(-DGLM_ENABLE_EXPERIMENTAL)

# --------------------------------------------------------------------
# |  Git                                                             |
# --------------------------------------------------------------------
find_package(Git REQUIRED)
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)

# --------------------------------------------------------------------
# |  Lua                                                             |
# --------------------------------------------------------------------
find_package(Lua51)
find_package(Lua52)
find_package(LuaJIT51 REQUIRED)
include_directories(${LUA_INCLUDE_DIR})
include_directories("/usr/include/lua5.1")

set(LUAWRAPPER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/luawrapper/include")
include_directories(${LUAWRAPPER_INCLUDE_DIR})

# --------------------------------------------------------------------
# |  Google Test                                                     |
# --------------------------------------------------------------------
set(GOOGLETEST_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/ext/googletest/googletest/include")
include_directories(${GOOGLETEST_INCLUDE_DIR})
include(CTest)

# --------------------------------------------------------------------
# |  qt-unix-signals                                                 |
# --------------------------------------------------------------------
set(QT_UNIX_SIGNALS_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/ext/qt-unix-signals")
include_directories(${QT_UNIX_SIGNALS_INCLUDE_DIR})

# --------------------------------------------------------------------
# |  Download Git submodules                                         |
# --------------------------------------------------------------------
if((NOT EXISTS GOOGLETEST_INCLUDE_DIR)
   OR (NOT EXISTS LUAWRAPPER_INCLUDE_DIR)
   OR (NOT EXISTS QT_UNIX_SIGNALS_INCLUDE_DIR))
  execute_process(COMMAND git submodule update --init --recursive
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# --------------------------------------------------------------------
# |  Source files                                                    |
# --------------------------------------------------------------------
# Add git submodules to build
add_subdirectory(${PROJECT_SOURCE_DIR}/ext/qt-unix-signals)
add_subdirectory(${PROJECT_SOURCE_DIR}/ext/googletest)

# Add all files recursively from src and include dirs
include_directories(include)
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
file(GLOB_RECURSE HEADER_FILES include/*.hpp)

# Read current git revision string
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/GitSHA1.cpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/GitSHA1.cpp" @ONLY)
list(APPEND SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/GitSHA1.cpp")

# Add QT5 resources
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
target_link_libraries(rafsine rafsine_lib QTSignal)
qt5_use_modules(rafsine Core Widgets Gui OpenGL)

# --------------------------------------------------------------------
# |  Unit tests                                                      |
# --------------------------------------------------------------------
file(GLOB_RECURSE UNIT_TEST_FILES tests/test_*.cpp)

set_source_files_properties(${UNIT_TEST_FILES}
                            PROPERTIES
                            CUDA_SOURCE_PROPERTY_FORMAT
                            OBJ)

cuda_add_executable(unit-tests
                    ${UNIT_TEST_FILES}
                    ${SOURCE_FILES}
                    ${MOC_HEADERS})

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

set(VOXMESH_DEMO tests/demo_voxelmesh.cpp)
set_source_files_properties(${VOXMESH_DEMO}
                            PROPERTIES
                            CUDA_SOURCE_PROPERTY_FORMAT
                            OBJ)
cuda_add_executable(voxmesh-demo ${VOXMESH_DEMO} ${SOURCE_FILES} ${MOC_HEADERS})
target_link_libraries(voxmesh-demo rafsine_lib)

# --------------------------------------------------------------------
# |  Post build move executable to parent directory                  |
# --------------------------------------------------------------------
add_custom_command(TARGET rafsine POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rafsine>
                           ${PROJECT_SOURCE_DIR})

# --------------------------------------------------------------------
# |  Doxygen                                                         |
# --------------------------------------------------------------------
find_package(Doxygen REQUIRED dot OPTIONAL_COMPONENTS mscgen dia)
if(DOXYGEN_FOUND AND DOXYGEN_ENABLED)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
                 ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doxygen ALL
                    COMMAND ${DOXYGEN_EXECUTABLE}
                            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile OUTPUT_QUIET
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMENT "Generating API documentation with Doxygen"
                    VERBATIM)
endif()

# --------------------------------------------------------------------
# |  Debug print                                                     |
# --------------------------------------------------------------------
message(STATUS "PROJECT_GIT_VERSION: ${GIT_SHA1}")

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

message(STATUS "LUAWRAPPER_INCLUDE_DIR: ${LUAWRAPPER_INCLUDE_DIR}")
message(STATUS "GOOGLETEST_INCLUDE_DIR: ${GOOGLETEST_INCLUDE_DIR}")

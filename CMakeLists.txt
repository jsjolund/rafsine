cmake_minimum_required(VERSION 3.18)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(DOXYGEN_ENABLED OFF)
set(BUILD_TESTS ON)
set(BUILD_DEMOS OFF)

# --------------------------------------------------------------------
# |  Compiler settings                                               |
# --------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TARGET_CUDA_ARCHITECTURES 61) # Target Nvidia GTX 1080 Ti

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.1/bin/nvcc") # Ubuntu 18.04
set(CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;-rdc=true")
set(CUDA_NVCC_FLAGS_DEBUG "-G")

# --------------------------------------------------------------------
# |  Use CCache if installed                                         |
# --------------------------------------------------------------------
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# --------------------------------------------------------------------
# |  Project                                                         |
# --------------------------------------------------------------------
project(rafsine LANGUAGES CXX CUDA)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake-modules")
set(FIND_LIBRARY_USE_LIB64_PATHS ON)
enable_testing()

# --------------------------------------------------------------------
# |  OpenMP                                                          |
# --------------------------------------------------------------------
find_package(OpenMP REQUIRED)

# --------------------------------------------------------------------
# |  CUDA                                                            |
# --------------------------------------------------------------------
find_package(CUDA 9.1 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
add_definitions(-DHAVE_CUDA)
set(CUDA_SEPARABLE_COMPILATION ON)

# Add missing libnvToolsExt
get_filename_component(CUDA_CUDART_LIBRARY_DIR "${CUDA_CUDART_LIBRARY}" PATH
                       CACHE)
find_library(
  CUDA_nvToolsExt_LIBRARY
  NAMES nvToolsExt nvtoolsext
  PATHS "${CUDA_CUDART_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}" ENV
        LD_LIBRARY_PATH
  PATH_SUFFIXES "lib64" "lib"
  DOC "Location of the NVIDIA Tools Extension library"
  NO_DEFAULT_PATH)

# --------------------------------------------------------------------
# |  OpenSceneGraph                                                  |
# --------------------------------------------------------------------
find_package(
  OSG
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
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# --------------------------------------------------------------------
# |  OpenGL                                                          |
# --------------------------------------------------------------------
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# --------------------------------------------------------------------
# |  Lua                                                             |
# --------------------------------------------------------------------
find_package(LuaJIT51 REQUIRED)
include_directories(${LUAJIT_INCLUDE_DIR})
set(LUAWRAPPER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/luawrapper/include")
include_directories(${LUAWRAPPER_INCLUDE_DIR})

# --------------------------------------------------------------------
# |  Google Test                                                     |
# --------------------------------------------------------------------
set(GOOGLETEST_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/googletest/googletest/include")
include_directories(${GOOGLETEST_INCLUDE_DIR})
include(CTest)

# --------------------------------------------------------------------
# |  qt-unix-signals                                                 |
# --------------------------------------------------------------------
set(QT_UNIX_SIGNALS_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/qt-unix-signals")
include_directories(${QT_UNIX_SIGNALS_INCLUDE_DIR})

# --------------------------------------------------------------------
# |  rapidcsv                                                        |
# --------------------------------------------------------------------
set(RAPIDCSV_PARSER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/rapidcsv/src")
include_directories(${RAPIDCSV_PARSER_INCLUDE_DIR})

# --------------------------------------------------------------------
# |  cpptoml                                                         |
# --------------------------------------------------------------------
set(CPPTOML_PARSER_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/cpptoml/include")
include_directories(${CPPTOML_PARSER_INCLUDE_DIR})

# --------------------------------------------------------------------
# |  pybind11                                                        |
# --------------------------------------------------------------------
find_package(PythonInterp 3.6 REQUIRED)
set(PYBIND11_PYTHON_VERSION 3.6)
set(PYBIND11_CPP_STANDARD -std=c++11)
set(PYBIND11_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/pybind11/include")
include_directories(${PYBIND11_INCLUDE_DIR})

# --------------------------------------------------------------------
# |  Cuda-Histogram                                                  |
# --------------------------------------------------------------------
set(CUDA_HISTOGRAM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/Cuda-Histogram")
include_directories(${CUDA_HISTOGRAM_INCLUDE_DIR})

# --------------------------------------------------------------------
# |  Git                                                             |
# --------------------------------------------------------------------
find_package(Git REQUIRED)
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)

# Update submodules as needed
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --remote --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(
        FATAL_ERROR
          "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules"
      )
    endif()
  endif()
endif()

# Add git submodules to build
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/googletest)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/qt-unix-signals)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/pybind11)

# --------------------------------------------------------------------
# |  Source files                                                    |
# --------------------------------------------------------------------

# Add all files recursively from src and include dirs
include_directories(include)
file(GLOB_RECURSE CXX_SOURCE_FILES src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCE_FILES src/*.cu)
file(GLOB_RECURSE HEADER_FILES include/*.hpp)

# Read current git revision string
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/GitSHA1.cpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/GitSHA1.cpp" @ONLY)
list(APPEND CXX_SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/GitSHA1.cpp")

set(SOURCE_FILES "")
list(APPEND SOURCE_FILES "${CXX_SOURCE_FILES}")
list(APPEND SOURCE_FILES "${CUDA_SOURCE_FILES}")

# Add QT5 resources
qt5_add_resources(RC_SRC "${PROJECT_SOURCE_DIR}/res.qrc")
qt5_wrap_cpp(MOC_HEADERS ${HEADER_FILES})

cuda_add_library(rafsine_lib SHARED ${SOURCE_FILES} ${MOC_HEADERS})

set_property(TARGET rafsine_lib PROPERTY CUDA_ARCHITECTURES
                                         ${TARGET_CUDA_ARCHITECTURES})

target_link_libraries(
  rafsine_lib
  -lcuda
  ${CUDA_nvToolsExt_LIBRARY}
  ${CUDA_LIBRARIES}
  ${LUAJIT_LIBRARY}
  ${OPENGL_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OSG_LIBRARY}
  ${OSGTEXT_LIBRARY}
  ${OSGDB_LIBRARY}
  ${OSGGA_LIBRARY}
  ${OSGUTIL_LIBRARY}
  ${OSGVIEWER_LIBRARY}
  ${OPENTHREADS_LIBRARY}
  Qt5::Core
  Qt5::Gui
  Qt5::OpenGL
  Qt5::Widgets
  OpenMP::OpenMP_CXX)

# --------------------------------------------------------------------
# |  Main executable                                                 |
# --------------------------------------------------------------------
cuda_add_executable(rafsine main.cpp ${SOURCE_FILES} ${MOC_HEADERS} ${RC_SRC})

target_link_libraries(rafsine rafsine_lib QTSignal)

set_property(TARGET rafsine PROPERTY CUDA_ARCHITECTURES
                                     ${TARGET_CUDA_ARCHITECTURES})

# --------------------------------------------------------------------
# |  Python  bindings                                                |
# --------------------------------------------------------------------
cuda_add_library(python_lbm MODULE python/wrapper/PythonClient.cpp)

set_target_properties(
  python_lbm
  PROPERTIES CXX_VISIBILITY_PRESET "hidden" PREFIX "${PYTHON_MODULE_PREFIX}"
             SUFFIX "${PYTHON_MODULE_EXTENSION}")

set_property(TARGET python_lbm PROPERTY CUDA_ARCHITECTURES
                                        ${TARGET_CUDA_ARCHITECTURES})

target_link_libraries(python_lbm rafsine_lib pybind11::module)

# --------------------------------------------------------------------
# |  Unit tests                                                      |
# --------------------------------------------------------------------
file(GLOB_RECURSE UNIT_TEST_FILES tests/test_*.cpp)
file(GLOB_RECURSE UNIT_TEST_HEADERS tests/test_cuda.hpp)

set_source_files_properties(${UNIT_TEST_FILES}
                            PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

cuda_add_executable(unit_tests ${UNIT_TEST_FILES} ${UNIT_TEST_HEADERS}
                    ${SOURCE_FILES} ${MOC_HEADERS})

set_property(TARGET unit_tests PROPERTY CUDA_ARCHITECTURES
                                        ${TARGET_CUDA_ARCHITECTURES})

target_link_libraries(unit_tests rafsine_lib gtest gtest_main)

if(NOT BUILD_TESTS)
  set_target_properties(unit_tests PROPERTIES EXCLUDE_FROM_ALL 1
                                              EXCLUDE_FROM_DEFAULT_BUILD 1)
else()
  add_test(tests unit_tests)
endif()

# --------------------------------------------------------------------
# |  Other executables                                               |
# --------------------------------------------------------------------
set(PARTITION_DEMO tests/demo_partition.cpp)
cuda_add_executable(partition-demo ${PARTITION_DEMO} ${SOURCE_FILES}
                    ${MOC_HEADERS})
set_property(TARGET partition-demo PROPERTY CUDA_ARCHITECTURES
                                            ${TARGET_CUDA_ARCHITECTURES})
target_link_libraries(partition-demo rafsine_lib)

# --------------------------------------------------------------------
set(VOXMESH_DEMO tests/demo_voxelmesh.cpp)
cuda_add_executable(voxmesh-demo ${VOXMESH_DEMO} ${SOURCE_FILES} ${MOC_HEADERS})
set_property(TARGET voxmesh-demo PROPERTY CUDA_ARCHITECTURES
                                          ${TARGET_CUDA_ARCHITECTURES})
target_link_libraries(voxmesh-demo rafsine_lib)

# --------------------------------------------------------------------
set(VOXSPHERE_DEMO tests/demo_voxelsphere.cpp)
cuda_add_executable(voxsphere-demo ${VOXSPHERE_DEMO} ${SOURCE_FILES}
                    ${MOC_HEADERS})
set_property(TARGET voxsphere-demo PROPERTY CUDA_ARCHITECTURES
                                            ${TARGET_CUDA_ARCHITECTURES})
target_link_libraries(voxsphere-demo rafsine_lib)

# --------------------------------------------------------------------
if(NOT BUILD_DEMOS)
  set_target_properties(
    partition-demo voxmesh-demo voxsphere-demo
    PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
endif()

# --------------------------------------------------------------------
# |  Doxygen                                                         |
# --------------------------------------------------------------------
find_package(Doxygen REQUIRED dot OPTIONAL_COMPONENTS mscgen dia)
if(DOXYGEN_FOUND AND DOXYGEN_ENABLED)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
                 ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  # Generate documentation
  add_custom_target(
    doxygen ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            OUTPUT_QUIET
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
  # Copy docs to parent folder
  add_custom_command(
    TARGET doxygen
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/docs/ ${PROJECT_SOURCE_DIR}/docs/)
endif()

# --------------------------------------------------------------------
# |  Post build move executable to parent directory                  |
# --------------------------------------------------------------------
add_custom_command(
  TARGET rafsine
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rafsine> ${PROJECT_SOURCE_DIR})

add_custom_command(
  TARGET python_lbm
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:python_lbm>
          "${PROJECT_SOURCE_DIR}/python/wrapper")

  
# --------------------------------------------------------------------
# |  Debug print                                                     |
# --------------------------------------------------------------------
message(STATUS "CMAKE_VERSION: ${CMAKE_VERSION}")
message(STATUS "CMAKE_SYSTEM: ${CMAKE_SYSTEM}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

message(STATUS "CCACHE_PATH: ${CCACHE_FOUND}")

message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
message(STATUS "CUDA_nvToolsExt_LIBRARY: ${CUDA_nvToolsExt_LIBRARY}")

message(STATUS "OSG_INCLUDE_DIR: ${OSG_INCLUDE_DIR}")
message(STATUS "OSG_LIBRARY: ${OSG_LIBRARY}")
message(STATUS "OPENTHREADS_LIBRARY: ${OPENTHREADS_LIBRARY}")

message(STATUS "OpenGL_OpenGL_FOUND: ${OpenGL_OpenGL_FOUND}")
message(STATUS "OPENGL_glu_LIBRARY: ${OPENGL_glu_LIBRARY}")
message(STATUS "OPENGL_glx_LIBRARY: ${OPENGL_glx_LIBRARY}")
message(STATUS "OPENGL_opengl_LIBRARY: ${OPENGL_opengl_LIBRARY}")
message(STATUS "OPENGL_INCLUDE_DIR: ${OPENGL_INCLUDE_DIR}")
message(STATUS "OPENGL_LIBRARIES: ${OPENGL_LIBRARIES}")

message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
message(STATUS "Boost_VERSION: ${Boost_VERSION}")

message(STATUS "LUAJIT_INCLUDE_DIR: ${LUAJIT_INCLUDE_DIR}")
message(STATUS "LUAJIT_LIBRARY: ${LUAJIT_LIBRARY}")

message(STATUS "PYTHON_VERSION_STRING: ${PYTHON_VERSION_STRING}")
message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_MODULE_EXTENSION: ${PYTHON_MODULE_EXTENSION}")

message(STATUS "PROJECT_VERSION: ${GIT_SHA1}")

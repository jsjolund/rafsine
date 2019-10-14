
#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/time.h>

#include <Eigen/Geometry>

#include <chrono>
#include <string>

#include "LbmFile.hpp"
#include "SimulationWorker.hpp"
#include "VoxelArray.hpp"

namespace py = pybind11;

struct PyBoundaryCondition {
  voxel_t m_id;
  VoxelType::Enum m_type;
  real m_temperature;
  float m_velocity[3];
  float m_normal[3];
  int m_rel_pos[3];

  explicit PyBoundaryCondition(BoundaryCondition bc)
      : m_id(bc.m_id), m_type(bc.m_type), m_temperature(bc.m_temperature) {
    std::copy(bc.m_velocity.data(), bc.m_velocity.data() + bc.m_velocity.size(),
              m_velocity);
    std::copy(bc.m_normal.data(), bc.m_normal.data() + bc.m_normal.size(),
              m_normal);
    std::copy(bc.m_rel_pos.data(), bc.m_rel_pos.data() + bc.m_rel_pos.size(),
              m_rel_pos);
  }
};

class Simulation {
 private:
  LbmFile m_lbmFile;
  SimulationWorker* m_simWorker;

 public:
  ~Simulation() { delete m_simWorker; }

  explicit Simulation(std::string lbmFilePath) : m_lbmFile(lbmFilePath) {
    int numDevices;
    CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
    assert(numDevices > 0);
    m_simWorker = new SimulationWorker(m_lbmFile, numDevices);
    m_simWorker->addAveragingObserver(new StdoutObserver());
  }

  std::vector<BoundaryCondition> get_boundary_conditions() {
    return *m_simWorker->getVoxels()->getBoundaryConditions();
  }

  // py::array_t<PyBoundaryCondition> get_boundary_conditions() {
  //   std::shared_ptr<BoundaryConditions> bcs =
  //       m_simWorker->getVoxels()->getBoundaryConditions();
  //   std::vector<PyBoundaryCondition> pbcs;
  //   for (BoundaryCondition bc : *bcs)
  //     pbcs.push_back(PyBoundaryCondition(bc));
  //   return py::array_t<PyBoundaryCondition>(pbcs.size(), pbcs.data());
  // }

  std::chrono::system_clock::time_point get_time() {
    timeval tv = m_simWorker->getSimulationTimer()->getTime();
    using namespace std::chrono;
    return system_clock::time_point{seconds{tv.tv_sec} +
                                    microseconds{tv.tv_usec}};
  }

  void run(float seconds) {
    unsigned int iterations = m_simWorker->getUnitConverter()->s_to_N(seconds);
    m_simWorker->run(iterations);
  }
};

PYBIND11_MODULE(python_lbm, m) {
  py::enum_<VoxelType::Enum>(m, "VoxelType.Enum")
      .value("EMPTY", VoxelType::Enum::EMPTY)
      .value("FLUID", VoxelType::Enum::FLUID)
      .value("WALL", VoxelType::Enum::WALL)
      .value("FREE_SLIP", VoxelType::Enum::FREE_SLIP)
      .value("INLET_CONSTANT", VoxelType::Enum::INLET_CONSTANT)
      .value("INLET_ZERO_GRADIENT", VoxelType::Enum::INLET_ZERO_GRADIENT)
      .value("INLET_RELATIVE", VoxelType::Enum::INLET_RELATIVE)
      .export_values();

  py::class_<BoundaryCondition>(m, "BoundaryCondition")
      .def_readwrite("id", &BoundaryCondition::m_id)
      .def_readwrite("type", &BoundaryCondition::m_type)
      .def_readwrite("temperature", &BoundaryCondition::m_temperature)
      .def_readwrite("velocity", &BoundaryCondition::m_velocity)
      .def_readwrite("normal", &BoundaryCondition::m_normal)
      .def_readwrite("rel_pos", &BoundaryCondition::m_rel_pos);

  // PYBIND11_NUMPY_DTYPE(PyBoundaryCondition, m_id, m_type, m_temperature,
  //                      m_velocity, m_normal, m_rel_pos);

  py::class_<Simulation>(m, "Simulation")
      .def(py::init<std::string>())
      .def("get_boundary_conditions", &Simulation::get_boundary_conditions)
      .def("get_time", &Simulation::get_time)
      .def("run", &Simulation::run);

  m.attr("__version__") = "dev";
}

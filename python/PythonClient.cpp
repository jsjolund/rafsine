
#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <string>

#include "LbmFile.hpp"
#include "SimulationWorker.hpp"
#include "VoxelArray.hpp"

namespace py = pybind11;

class Simulation {
 private:
  LbmFile m_lbmFile;
  SimulationWorker* m_simWorker;
  ListAveraging* m_avgs;

 public:
  ~Simulation() {
    delete m_simWorker;
    delete m_avgs;
  }

  explicit Simulation(std::string lbmFilePath) {
    int numDevices;
    CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
    assert(numDevices > 0);

    m_lbmFile = LbmFile(lbmFilePath);
    m_simWorker = new SimulationWorker(m_lbmFile, numDevices);
    m_avgs = new ListAveraging();
    m_simWorker->addAveragingObserver(m_avgs);
  }

  void set_time_averaging_period(float seconds) {
    m_simWorker->setAveragingPeriod(seconds);
  }

  py::list get_average_names() {
    const AverageMatrix& mat = m_avgs->getAverages();
    py::list result;
    result.append("time");
    for (std::string col : mat.m_columns) result.append(col);
    return result;
  }

  py::list get_averages() {
    const AverageMatrix& mat = m_avgs->getAverages();
    py::list result;
    for (AverageData data : mat.m_rows) {
      py::array_t<Average> measurements(
          py::buffer_info(data.m_measurements.data(), sizeof(Average),
                          py::format_descriptor<Average>::format(), 1,
                          std::vector<size_t>{data.m_measurements.size()},
                          std::vector<size_t>{sizeof(Average)}));
      result.append(py::make_tuple(data.m_time, *measurements));
    }
    return result;
  }

  std::vector<BoundaryCondition> get_boundary_conditions() {
    return *m_simWorker->getVoxels()->getBoundaryConditions();
  }

  real get_time_step() { return m_simWorker->getUnitConverter()->C_T(); }

  std::chrono::system_clock::time_point get_time() {
    timeval tv = m_simWorker->getSimulationTimer()->getTime();
    std::chrono::system_clock::time_point tp;
    timevalToTimepoint(tv, &tp);
    return tp;
  }

  void run(float seconds) {
    unsigned int iterations = m_simWorker->getUnitConverter()->s_to_N(seconds);
    m_simWorker->run(iterations);
  }
};

PYBIND11_MODULE(python_lbm, m) {
  m.doc() = "LUA LBM GPU Leeds Luleå 2019";
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

  PYBIND11_NUMPY_DTYPE(Average, temperature, velocity, flow);

  py::class_<Simulation>(m, "Simulation")
      .def(py::init<std::string>(), "Load a simulation from lbm file")
      .def("get_boundary_conditions", &Simulation::get_boundary_conditions,
           "List the current boundary conditions")
      .def("set_time_averaging_period", &Simulation::set_time_averaging_period,
           "Set the time averaging period in seconds")
      .def("get_average_names", &Simulation::get_average_names,
           "List the names of the time averages for measurement areas")
      .def("get_averages", &Simulation::get_averages,
           "List the time averages of measurement areas")
      .def("get_time", &Simulation::get_time,
           "Get current time in the simulation domain")
      .def("get_time_step", &Simulation::get_time_step,
           "Get the seconds of simulated time for one discrete time step")
      .def("run", &Simulation::run,
           "Run the simulation for a number of seconds of simulated time");

  m.attr("__version__") = "dev";
}

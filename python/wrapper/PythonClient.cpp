
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <chrono>
#include <string>

#include "BasicTimer.hpp"
#include "LbmFile.hpp"
#include "SimulationWorker.hpp"
#include "VoxelArray.hpp"

namespace py = pybind11;

class PythonClient {
 private:
  LbmFile m_lbmFile;
  SimulationWorker* m_simWorker;
  ListAveraging* m_avgs;

 public:
  ~PythonClient() {
    delete m_simWorker;
    delete m_avgs;
  }

  explicit PythonClient(std::string lbmFilePath) {
    int nd;
    CUDA_RT_CALL(cudaGetDeviceCount(&nd));
    assert(nd > 0);

    m_lbmFile = LbmFile(lbmFilePath);
    m_simWorker = new SimulationWorker(m_lbmFile, nd);
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

  real_t get_time_step() { return m_simWorker->getUnitConverter()->C_T(); }

  sim_clock_t::time_point get_time() {
    return m_simWorker->getSimulationTimer()->getTime();
  }

  void upload_boundary_conditions() { m_simWorker->uploadBCs(); }

  void set_boundary_condition(std::string name,
                              float temperature,
                              float vol_flow) {
    std::unordered_set<VoxelQuad> quads =
        m_simWorker->getVoxels()->getQuadsByName(name);
    std::shared_ptr<UnitConverter> uc = m_simWorker->getUnitConverter();
    std::shared_ptr<BoundaryConditions> bcs =
        m_simWorker->getVoxels()->getBoundaryConditions();
    for (VoxelQuad quad : quads) {
      BoundaryCondition* bc = &(bcs->at(quad.m_bc.m_id));
      bc->setTemperature(*uc, temperature);
      bc->setFlow(*uc, vol_flow, quad.getAreaDiscrete(*uc));
    }
  }

  std::vector<std::string> get_boundary_condition_names() {
    return m_simWorker->getVoxels()->getGeometryNames();
  }

  std::vector<int> get_boundary_condition_ids_from_name(std::string name) {
    return m_simWorker->getVoxels()->getIdsByName(name);
  }

  void run(float seconds) {
    unsigned int iterations = m_simWorker->getUnitConverter()->s_to_N(seconds);
    m_simWorker->run(iterations);
  }
};

PYBIND11_MODULE(python_lbm, m) {
  m.doc() = "LBM GPU Leeds Luleå 2019";
  py::enum_<VoxelType::Enum>(m, "VoxelType", py::module_local())
      .value("EMPTY", VoxelType::Enum::EMPTY)
      .value("FLUID", VoxelType::Enum::FLUID)
      .value("WALL", VoxelType::Enum::WALL)
      .value("FREE_SLIP", VoxelType::Enum::FREE_SLIP)
      .value("INLET_CONSTANT", VoxelType::Enum::INLET_CONSTANT)
      .value("INLET_ZERO_GRADIENT", VoxelType::Enum::INLET_ZERO_GRADIENT)
      .value("INLET_RELATIVE", VoxelType::Enum::INLET_RELATIVE);

  py::class_<BoundaryCondition>(m, "BoundaryCondition")
      .def_readwrite("id", &BoundaryCondition::m_id)
      .def_readwrite("type", &BoundaryCondition::m_type)
      .def_readwrite("temperature", &BoundaryCondition::m_temperature)
      .def_readwrite("velocity", &BoundaryCondition::m_velocity)
      .def_readwrite("normal", &BoundaryCondition::m_normal)
      .def_readwrite("rel_pos", &BoundaryCondition::m_rel_pos)
      .def_readwrite("tau1", &BoundaryCondition::m_tau1)
      .def_readwrite("tau2", &BoundaryCondition::m_tau2)
      .def_readwrite("lambda1", &BoundaryCondition::m_lambda);

  PYBIND11_NUMPY_DTYPE(Average, temperature, velocity, flow);

  py::class_<PythonClient>(m, "PythonClient")
      .def(py::init<std::string>())
      .def("get_boundary_conditions", &PythonClient::get_boundary_conditions)
      .def("get_boundary_condition_names",
           &PythonClient::get_boundary_condition_names)
      .def("get_boundary_condition_ids_from_name",
           &PythonClient::get_boundary_condition_ids_from_name)
      .def("set_boundary_condition", &PythonClient::set_boundary_condition)
      .def("set_time_averaging_period",
           &PythonClient::set_time_averaging_period)
      .def("get_average_names", &PythonClient::get_average_names)
      .def("get_averages", &PythonClient::get_averages)
      .def("get_time", &PythonClient::get_time)
      .def("get_time_step", &PythonClient::get_time_step)
      .def("upload_boundary_conditions",
           &PythonClient::upload_boundary_conditions)
      .def("run", &PythonClient::run);

  m.attr("__version__") = "dev";
}

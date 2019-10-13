
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/time.h>

#include <LbmFile.hpp>
#include <SimulationWorker.hpp>
#include <chrono>
#include <string>

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
    m_simWorker = new SimulationWorker(m_lbmFile, numDevices, 10);
    m_simWorker->addAveragingObserver(new StdoutObserver());
  }

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

namespace py = pybind11;

PYBIND11_MODULE(python_lbm, m) {
  py::class_<Simulation>(m, "Simulation")
      .def(py::init<std::string>())
      .def("run", &Simulation::run)
      .def("get_time", &Simulation::get_time);

  m.attr("__version__") = "dev";
}


#include <pybind11/pybind11.h>

#include <string>

#include <LbmFile.hpp>
#include <SimulationWorker.hpp>

class Client {
 private:
  LbmFile m_lbmFile;
  SimulationWorker *m_simWorker;

 public:
  explicit Client(std::string lbmFilePath) : m_lbmFile(lbmFilePath) {
    int numDevices;
    CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
    assert(numDevices > 0);
    m_simWorker = new SimulationWorker(m_lbmFile, numDevices, 10);
    m_simWorker->addAveragingObserver(new StdoutObserver());
    m_simWorker->run();
  }
  std::string getTitle() { return m_lbmFile.getTitle(); }
};

namespace py = pybind11;

PYBIND11_MODULE(python_lbm, m) {
  py::class_<Client>(m, "Client")
      .def(py::init<std::string>())
      .def("getTitle", &Client::getTitle);

  m.attr("__version__") = "dev";
}

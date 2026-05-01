#include "LifNetConstructor.h"
#include "lifnet.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pybnn, m) {
  py::class_<LifNet>(m, "LifNet")
      .def(py::init<std::string>())
      .def("AddSensoryNeuron", &LifNet::AddSensoryNeuron)
      .def("AddMotorNeuron", &LifNet::AddMotorNeuron)
      .def("AddBiSensoryNeuron", &LifNet::AddBiSensoryNeuron)
      .def("AddBiMotorNeuron", &LifNet::AddBiMotorNeuron)
      .def("Reset", &LifNet::Reset)
      .def("DumpClear", &LifNet::DumpClear)
      .def("DumpState", &LifNet::DumpState)
      .def("AddNoise", &LifNet::AddNoise)
      .def("AddNoiseSigma", &LifNet::AddNoiseSigma)
      .def("AddNoiseVleak", &LifNet::AddNoiseVleak)
      .def("AddNoiseGleak", &LifNet::AddNoiseGleak)
      .def("AddNoiseCm", &LifNet::AddNoiseCm)
      .def("UndoNoise", &LifNet::UndoNoise)
      .def("CommitNoise", &LifNet::CommitNoise)
      .def("SeedRandomNumberGenerator", &LifNet::SeedRandomNumberGenerator)
      .def("WriteToFile", &LifNet::WriteToFile)
      .def("Update", &LifNet::Update);

  py::class_<LifNetConstructor>(m, "LifNetConstructor")
      .def(py::init<int>())
      .def("AddExcitatorySynapse", &LifNetConstructor::AddExcitatorySynapse)
      .def("AddInhibitorySynapse", &LifNetConstructor::AddInhibitorySynapse)
      .def("AddGapJunction", &LifNetConstructor::AddGapJunction)
      .def("AddConstNeuron", &LifNetConstructor::AddConstNeuron)
      .def("CountSynapses", &LifNetConstructor::CountSynapses)
      .def("WriteToFile", &LifNetConstructor::WriteToFile);
}

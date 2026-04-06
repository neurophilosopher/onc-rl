#ifndef LIF_NET_H
#define LIF_NET_H

#include "LifNetConstructor.h"
#include "NeuralInterface.h"
#include "neural_net_serializer.h"
#include "neural_network.h"
#include <random>
#include <string>
#include <vector>

class LifNet {
private:
  NeuralNetwork *nn;
  std::vector<NeuralInterfaceInput> inputs;
  std::vector<NeuralInterfaceOutput> outputs;
  std::vector<NeuralBiInterfaceInput> biInputs;
  std::vector<NeuralBiInterfaceOutput> biOutputs;
  std::mt19937 gen;
  std::vector<float *> parameters;
  std::vector<float> backupValues;

  std::vector<float *> sigmaParameters;
  std::vector<float> sigmaBackupValues;

  std::vector<float *> vleakParameters;
  std::vector<float> vleakBackupValues;

  std::vector<float *> gleakParameters;
  std::vector<float> gleakBackupValues;

  std::vector<float *> cmParameters;
  std::vector<float> cmBackupValues;
  float totalTime;

public:
  LifNet(std::string);
  ~LifNet();
  void AddSensoryNeuron(int indx, float input_bound);
  void AddMotorNeuron(int indx, float output_bound);
  void AddBiMotorNeuron(int indxPos, int indxNeg, float min_value,
                        float max_value);
  void AddBiSensoryNeuron(int indxPos, int indxNeg, float min_value,
                          float max_value);
  std::vector<float> Update(const std::vector<float> &inputsArr, float deltaT,
                            int simulationSteps);
  void WriteToFile(std::string);
  void AddNoise(float variance, int samples);
  void AddNoiseSigma(float variance, int samples);
  void AddNoiseVleak(float variance, int samples);
  void AddNoiseGleak(float variance, int samples);
  void AddNoiseCm(float variance, int samples);
  void UndoNoise();
  void CommitNoise();
  void SeedRandomNumberGenerator(int seed);
  void Reset();
  void DumpState(std::string filename);
  void DumpClear(std::string filename);
};
#endif /* end of include guard: LIF_NET_H */

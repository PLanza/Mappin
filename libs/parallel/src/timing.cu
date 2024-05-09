#include "../include/parallel/timing.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

struct PrivateTiming {
  cudaEvent_t start;
  cudaEvent_t stop;
};

// default constructor
Timing::Timing() { privateTiming = new PrivateTiming; }

// default destructor
Timing::~Timing() {}

void Timing::StartCounter() {
  cudaEventCreate(&((*privateTiming).start));
  cudaEventCreate(&((*privateTiming).stop));
  cudaEventRecord((*privateTiming).start, 0);
}

void Timing::StartCounterFlags() {
  int eventflags = cudaEventBlockingSync;

  cudaEventCreateWithFlags(&((*privateTiming).start), eventflags);
  cudaEventCreateWithFlags(&((*privateTiming).stop), eventflags);
  cudaEventRecord((*privateTiming).start, 0);
}

// Gets the counter in ms
float Timing::GetCounter() {
  float time;
  cudaEventRecord((*privateTiming).stop, 0);
  cudaEventSynchronize((*privateTiming).stop);
  cudaEventElapsedTime(&time, (*privateTiming).start, (*privateTiming).stop);
  return time;
}


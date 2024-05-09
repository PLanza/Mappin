#ifndef __MAPPIN_PARALLEL_TIMING__
#define __MAPPIN_PARALLEL_TIMING__

struct PrivateTiming;

class Timing {
private:
  PrivateTiming *privateTiming;

public:
  Timing();

  ~Timing();

  void StartCounter();
  void StartCounterFlags();

  float GetCounter();

}; // TimingCPU class

#endif

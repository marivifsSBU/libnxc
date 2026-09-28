#ifndef DEFAULTS_H
#define DEFAULTS_H

const int DEFAULT_CODE=0;
const int SIESTA_GRID_CODE=1;
const int SIESTA_ATOMIC_CODE=2;
const int CP2K_CODE=0;

class Defaults {
  static Defaults *instance;
  Defaults(){};
  public:
    int edens, add, cuda, gamma, interleaved, tau_w_cap;
    double rho_threshold, sigma_threshold, spin_threshold, tau_threshold;
    static Defaults *getInstance() {
      if (!instance)
        instance = new Defaults;
      instance->edens = 1;
      instance->add = 0;
      instance->cuda = 0;
      instance->gamma = 0;
      instance->interleaved = 0;
      instance->tau_w_cap = 1;
      instance->rho_threshold = 1e-10;
      instance->sigma_threshold = 1e-40;
      instance->spin_threshold = 1e-15;
      instance->tau_threshold = 1e-20;
      return instance;
    }
    void setDefault(const int code);
    void useCuda();
  };
extern Defaults *defaults;

#endif

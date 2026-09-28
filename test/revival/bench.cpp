// Throughput of the C++ API: bench <model> <level> <nspin> <npoints> [repeats]
#include "nxc.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

int main(int argc, char **argv){
  if (argc < 5){ std::fprintf(stderr, "usage: bench model level nspin npoints [repeats]\n"); return 2; }
  std::string model = argv[1];
  int level = std::atoi(argv[2]), ns = std::atoi(argv[3]), np = std::atoi(argv[4]);
  int reps = argc > 5 ? std::atoi(argv[5]) : 5;
  int nsig = ns == 2 ? 3 : 1;
  std::mt19937_64 g(1);
  std::uniform_real_distribution<double> u(0, 1);
  const double kf = std::cbrt(3 * M_PI * M_PI), cf = 0.3 * std::pow(3 * M_PI * M_PI, 2.0 / 3);
  std::vector<double> rho(ns * np), sigma(nsig * np), lapl(ns * np, 0.0), tau(ns * np);
  for (int i = 0; i < np; ++i){
    double r = std::pow(10.0, -4 + 5 * u(g)), s = 3 * u(g), a = 2 * u(g);
    double sig = std::pow(s * 2 * kf * std::pow(r, 4.0 / 3), 2), t = sig / (8 * r) + a * cf * std::pow(r, 5.0 / 3);
    for (int k = 0; k < ns; ++k){ rho[k * np + i] = r / ns; tau[k * np + i] = t / ns; }
    for (int k = 0; k < nsig; ++k) sigma[k * np + i] = sig / (ns * ns);
  }
  std::vector<double> exc(np), vrho(ns * np), vsigma(nsig * np), vlapl(ns * np), vtau(ns * np);
  nxc_func_type p; func_param fp;
  if (nxc_func_init(&p, model, fp, ns) != 0) return 1;
  double best = 1e30;
  for (int r = 0; r < reps; ++r){
    auto t0 = std::chrono::steady_clock::now();
    if (level == 1) nxc_gga_exc_vxc(&p, np, rho.data(), sigma.data(), exc.data(), vrho.data(), vsigma.data());
    else nxc_mgga_exc_vxc(&p, np, rho.data(), sigma.data(), lapl.data(), tau.data(), exc.data(), vrho.data(),
                          vsigma.data(), vlapl.data(), vtau.data());
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    best = std::min(best, dt);
  }
  std::printf("%-28s nspin=%d np=%d: %.3f s per call, %.2f Mpoints/s\n",
              model.substr(model.find_last_of('/') + 1).c_str(), ns, np, best, np / best / 1e6);
  return 0;
}

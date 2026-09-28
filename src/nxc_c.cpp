// C bindings used by the Libxc hook (src/libxc/*.c).  NOTE: that hook targets the
// Libxc 4.3.4 internals (functional info struct, work-function signatures, autotools
// lists) and is not compatible with Libxc >= 5; it is kept for reference and untested.
#include <cstdlib>
#include <iostream>
#include <memory>
#include "nxc_c.h"
#include "nxc.h"
#include "funcs.h"
nxc_func_type nxc_func_c;

namespace {
template <typename F>
void or_abort(const char *where, F f){
  try{
    f();
  }catch (const std::exception &e){
    std::cerr << "libnxc (" << where << "): " << e.what() << std::endl;
    std::abort();
  }
}
}

void nxc_c_lda_exc_vxc(int * xc_func, int np, double rho[],
  double exc [], LDA_OUT_PARAMS_NOEXC(double *)){
  or_abort("lda", [&]{ nxc_lda_exc_vxc(&nxc_func_c, np, rho, exc, vrho); });
}

void nxc_c_gga_exc_vxc(int * xc_func, int np, double rho[], double sigma[],
  double exc [], GGA_OUT_PARAMS_NOEXC(double *)){
  or_abort("gga", [&]{ nxc_gga_exc_vxc(&nxc_func_c, np, rho, sigma, exc, vrho, vsigma); });
}

void nxc_c_mgga_exc_vxc(int * xc_func, int np, double rho[], double sigma[], double lapl[],
  double tau[], double exc [], MGGA_OUT_PARAMS_NOEXC(double *)){
  or_abort("mgga", [&]{
    nxc_mgga_exc_vxc(&nxc_func_c, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl_rho, vtau); });
}

int nxc_c_xc_init(int xc_func, int nspin){
  func_param fp;
  fp.interleaved = 1;          // Libxc passes rho[ip*nspin + is]
  return nxc_func_init(&nxc_func_c, funcs.at(xc_func), fp, nspin);
}

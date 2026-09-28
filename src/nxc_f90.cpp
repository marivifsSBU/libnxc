#include "nxc_f90.h"
#include "nxc.h"
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include "nxc_func.h"

namespace {
nxc_func_type nxc_func;
bool func_set = false;

// Run f; return 0 on success, 1 on a libnxc/torch error (message on stderr).
int guarded(const char *where, const std::function<void()> &f){
  if (!func_set){
    std::cerr << "libnxc (" << where << "): no functional initialized" << std::endl;
    return 2;
  }
  try{
    f();
    return 0;
  }catch (const std::exception &e){
    std::cerr << "libnxc (" << where << "): " << e.what() << std::endl;
    return 1;
  }
}

// Legacy entry points: callers do not always pass ierr, so errors stop the program.
void legacy_check(int ierr){
  if (ierr != 0){
    std::cerr << "libnxc: stopping (legacy Fortran interface has no error return)" << std::endl;
    std::exit(1);
  }
}

std::string fstring(const char *s, int len){
  std::string out(s, len > 0 ? len : 0);
  size_t end = out.find_last_not_of(" \t");
  return end == std::string::npos ? "" : out.substr(0, end + 1);
}
}

extern "C"{

/* ---------------- checked interface (Fortran module `nxc`, see nxc_mod.f90) ---------------- */

void nxcf_set_layout(int interleaved){ defaults->interleaved = interleaved; }
void nxcf_set_rho_threshold(double threshold){ defaults->rho_threshold = threshold; }

void nxcf_func_init(const char *modelpath, int pathlen, int nspin, int *ierr){
  func_param fp;
  func_set = (nxc_func_init(&nxc_func, fstring(modelpath, pathlen), fp, nspin) == 0);
  *ierr = func_set ? 0 : 1;
}

void nxcf_func_get_family(int *family){
  *family = func_set ? nxc_func_get_family(&nxc_func) : -1;
}

void nxcf_lda_exc_vxc(int np, double rho[], double exc[], double vrho[], int *ierr){
  *ierr = guarded("lda_exc_vxc", [&]{ nxc_lda_exc_vxc(&nxc_func, np, rho, exc, vrho); });
}

void nxcf_gga_exc_vxc(int np, double rho[], double sigma[], double exc[],
                      double vrho[], double vsigma[], int *ierr){
  *ierr = guarded("gga_exc_vxc", [&]{ nxc_gga_exc_vxc(&nxc_func, np, rho, sigma, exc, vrho, vsigma); });
}

void nxcf_mgga_exc_vxc(int np, double rho[], double sigma[], double lapl[], double tau[],
                       double exc[], double vrho[], double vsigma[], double vlapl[], double vtau[],
                       int *ierr){
  *ierr = guarded("mgga_exc_vxc", [&]{
    nxc_mgga_exc_vxc(&nxc_func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau); });
}

/* ---------------- legacy interface (implicit Fortran interfaces, used by the SIESTA patch) ---------------- */

void nxc_f90_set_code_(int * code){
  nxc_set_code(*code);
}
void nxc_f90_use_cuda_(){
  nxc_use_cuda();
}
void nxc_f90_cuda_available(int * available){
  *available = nxc_cuda_available();
}
void nxc_f90_atmfunc_init_(double  pos[], int * nua, double  cell[], int  grid[], int isa[],
            char symbols[], int * ns, char  modelpath[], int * pathlen, int myBox[], int* ierr){

    std::string modeldir(modelpath);
    modeldir = modeldir.erase(*pathlen, std::string::npos);
    func_param fp = {pos, *nua, cell, grid, isa, symbols, *ns, myBox};
    func_set = (nxc_func_init(&nxc_func, modeldir, fp) == 0);
    legacy_check(func_set ? 0 : 1);
}

void nxc_f90_func_init_(char  modelpath[], int * pathlen, int * ierr){
    int err;
    nxcf_func_init(modelpath, *pathlen, NXC_UNPOLARIZED, &err);
    legacy_check(err);
}

void nxc_f90_func_init_nspin_(char  modelpath[], int * pathlen, int * nspin, int * ierr){
    nxcf_func_init(modelpath, *pathlen, *nspin, ierr);
}

void nxc_f90_lda_exc_vxc_(int* np, double rho[], double exc [], double vrho[], int* ierr) {
  int err;
  nxcf_lda_exc_vxc(*np, rho, exc, vrho, &err);
  legacy_check(err);
}

void nxc_f90_lda_exc_vxc_fs_(int* np, double rho[], double exc[], double vrho[],
                            double forces[], double stress[], int* ierr) {
  legacy_check(guarded("lda_exc_vxc_fs", [&]{
    nxc_lda_exc_vxc_fs(&nxc_func, *np, rho, exc, vrho, forces, stress); }));
}

void nxc_f90_gga_exc_vxc_(int* np, double rho[], double sigma[], double exc [],
    double vrho[], double vsigma[], int* ierr) {
  int err;
  nxcf_gga_exc_vxc(*np, rho, sigma, exc, vrho, vsigma, &err);
  legacy_check(err);
}

void nxc_f90_mgga_exc_vxc_(int* np, double rho[], double sigma[], double lapl[], double tau[],
   double exc [], double vrho[], double vsigma[], double vlapl[], double vtau[], int* ierr) {
  int err;
  nxcf_mgga_exc_vxc(*np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau, &err);
  legacy_check(err);
}

void nxc_f90_func_get_family_(int * family){
  nxcf_func_get_family(family);
}
void nxc_f90_func_get_family_from_path_(char modelpath [], int * pathlen, int * family){
  try{
    *family = nxc_func_get_family_from_path(fstring(modelpath, *pathlen));
  }catch (const std::exception &e){
    std::cerr << "libnxc: " << e.what() << std::endl;
    *family = -1;
  }
}
}

#ifndef NXCF_H
#define NXCF_H
/*
 * Fortran interfaces.
 *
 * Array layout (both interfaces): np grid points, spin/components blocked with the grid
 * index fastest, i.e. Fortran rho(np, nspin), sigma(np, 1|3), tau(np, nspin); sigma holds
 * (aa, ab, bb) for nspin = 2.  nxcf_set_layout(1) switches to the Libxc interleaved layout.
 *
 * nxcf_*: checked interface (ierr = 0 ok, 1 libnxc/torch error, 2 no functional),
 *         bound in the Fortran module `nxc` (src/nxc_mod.f90).
 * nxc_f90_*_: legacy implicit-interface entry points (trailing underscore dropped when
 *         called from Fortran, e.g. `call nxc_f90_gga_exc_vxc(...)`).  Errors stop the
 *         program, since legacy callers do not always pass ierr.
 */
extern "C"{

void nxcf_set_layout(int interleaved);
void nxcf_set_rho_threshold(double threshold);
void nxcf_func_init(const char *modelpath, int pathlen, int nspin, int *ierr);
void nxcf_func_get_family(int *family);
void nxcf_lda_exc_vxc(int np, double rho[], double exc[], double vrho[], int *ierr);
void nxcf_gga_exc_vxc(int np, double rho[], double sigma[], double exc[],
                      double vrho[], double vsigma[], int *ierr);
void nxcf_mgga_exc_vxc(int np, double rho[], double sigma[], double lapl[], double tau[],
                       double exc[], double vrho[], double vsigma[], double vlapl[], double vtau[],
                       int *ierr);

void nxc_f90_set_code_(int * code);
void nxc_f90_use_cuda_();
void nxc_f90_cuda_available(int * available);

void nxc_f90_atmfunc_init_(double  pos[], int * nua, double  cell[], int  grid[], int isa[],
            char symbols[], int * ns, char  modelpath[], int * pathlen, int myBox[], int* ierr);

/** Unpolarized grid functional (legacy); see nxc_f90_func_init_nspin_ for spin. */
void nxc_f90_func_init_(char  modelpath[], int * pathlen, int * ierr);
void nxc_f90_func_init_nspin_(char  modelpath[], int * pathlen, int * nspin, int * ierr);

void nxc_f90_lda_exc_vxc_(int* np, double rho[], double exc [], double vrho[], int* ierr);
void nxc_f90_lda_exc_vxc_fs_(int* np, double rho[], double exc[], double vrho[],
                            double forces[], double stress[], int* ierr);
void nxc_f90_gga_exc_vxc_(int* np, double rho[], double sigma[], double exc [],
    double vrho[], double vsigma[], int* ierr);
void nxc_f90_mgga_exc_vxc_(int* np, double rho[], double sigma[], double lapl[], double tau[],
   double exc [], double vrho[], double vsigma[], double vlapl[], double vtau[], int* ierr);

void nxc_f90_func_get_family_(int * family);
void nxc_f90_func_get_family_from_path_(char modelpath [], int * pathlen, int * family);
}
#endif

!> Fortran interface to libnxc grid functionals (LDA / GGA / meta-GGA).
!>
!> Arrays: np grid points, grid index fastest (rho(np, nspin), sigma(np, 1|3) with
!> (aa, ab, bb) columns for nspin = 2, lapl/tau(np, nspin)); potentials have the shape
!> of the corresponding input.  exc is the energy per particle, E = sum w exc (rho_a+rho_b).
!> ierr: 0 ok, 1 libnxc/torch error (message on stderr), 2 no functional initialized.
!>
!>   use nxc
!>   call nxc_init('/path/to/GGA_XC_MYCLONE', 2, ierr)      ! directory containing 'xc'
!>   call nxc_gga(np, rho, sigma, exc, vrho, vsigma, ierr)
module nxc
  use, intrinsic :: iso_c_binding, only: c_int, c_double, c_char
  implicit none
  private
  public :: nxc_init, nxc_family, nxc_lda, nxc_gga, nxc_mgga, nxc_set_layout, &
            nxc_set_rho_threshold, NXC_LDA_TYPE, NXC_GGA_TYPE, NXC_MGGA_TYPE

  integer, parameter :: NXC_LDA_TYPE = 1, NXC_GGA_TYPE = 2, NXC_MGGA_TYPE = 4

  interface
    subroutine c_set_layout(interleaved) bind(C, name='nxcf_set_layout')
      import :: c_int
      integer(c_int), value :: interleaved
    end subroutine
    subroutine c_set_rho_threshold(thr) bind(C, name='nxcf_set_rho_threshold')
      import :: c_double
      real(c_double), value :: thr
    end subroutine
    subroutine c_func_init(path, pathlen, nspin, ierr) bind(C, name='nxcf_func_init')
      import :: c_int, c_char
      character(kind=c_char), intent(in) :: path(*)
      integer(c_int), value :: pathlen, nspin
      integer(c_int), intent(out) :: ierr
    end subroutine
    subroutine c_func_get_family(family) bind(C, name='nxcf_func_get_family')
      import :: c_int
      integer(c_int), intent(out) :: family
    end subroutine
    subroutine c_lda(np, rho, exc, vrho, ierr) bind(C, name='nxcf_lda_exc_vxc')
      import :: c_int, c_double
      integer(c_int), value :: np
      real(c_double), intent(in) :: rho(*)
      real(c_double), intent(inout) :: exc(*), vrho(*)
      integer(c_int), intent(out) :: ierr
    end subroutine
    subroutine c_gga(np, rho, sigma, exc, vrho, vsigma, ierr) bind(C, name='nxcf_gga_exc_vxc')
      import :: c_int, c_double
      integer(c_int), value :: np
      real(c_double), intent(in) :: rho(*), sigma(*)
      real(c_double), intent(inout) :: exc(*), vrho(*), vsigma(*)
      integer(c_int), intent(out) :: ierr
    end subroutine
    subroutine c_mgga(np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau, ierr) &
        bind(C, name='nxcf_mgga_exc_vxc')
      import :: c_int, c_double
      integer(c_int), value :: np
      real(c_double), intent(in) :: rho(*), sigma(*), lapl(*), tau(*)
      real(c_double), intent(inout) :: exc(*), vrho(*), vsigma(*), vlapl(*), vtau(*)
      integer(c_int), intent(out) :: ierr
    end subroutine
  end interface

contains

  !> Load a model (shipped name or directory containing 'xc'); nspin = 1 or 2.
  subroutine nxc_init(path, nspin, ierr)
    character(len=*), intent(in) :: path
    integer, intent(in) :: nspin
    integer, intent(out) :: ierr
    integer(c_int) :: cerr
    call c_func_init(path, int(len_trim(path), c_int), int(nspin, c_int), cerr)
    ierr = cerr
  end subroutine

  !> NXC_LDA_TYPE, NXC_GGA_TYPE, NXC_MGGA_TYPE, 5 (atomic) or -1 (none loaded)
  integer function nxc_family()
    integer(c_int) :: f
    call c_func_get_family(f)
    nxc_family = f
  end function

  !> 0: blocked arrays (default, rho(np, nspin)); 1: Libxc interleaved (rho(nspin, np)).
  !> Applies to functionals initialized afterwards.
  subroutine nxc_set_layout(interleaved)
    integer, intent(in) :: interleaved
    call c_set_layout(int(interleaved, c_int))
  end subroutine

  !> Points with rho_a + rho_b <= thr get exc = v = 0 (default 1e-10); applies to
  !> functionals initialized afterwards.
  subroutine nxc_set_rho_threshold(thr)
    real(c_double), intent(in) :: thr
    call c_set_rho_threshold(thr)
  end subroutine

  subroutine nxc_lda(np, rho, exc, vrho, ierr)
    integer, intent(in) :: np
    real(c_double), intent(in) :: rho(*)
    real(c_double), intent(inout) :: exc(*), vrho(*)
    integer, intent(out) :: ierr
    integer(c_int) :: cerr
    call c_lda(int(np, c_int), rho, exc, vrho, cerr)
    ierr = cerr
  end subroutine

  subroutine nxc_gga(np, rho, sigma, exc, vrho, vsigma, ierr)
    integer, intent(in) :: np
    real(c_double), intent(in) :: rho(*), sigma(*)
    real(c_double), intent(inout) :: exc(*), vrho(*), vsigma(*)
    integer, intent(out) :: ierr
    integer(c_int) :: cerr
    call c_gga(int(np, c_int), rho, sigma, exc, vrho, vsigma, cerr)
    ierr = cerr
  end subroutine

  subroutine nxc_mgga(np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau, ierr)
    integer, intent(in) :: np
    real(c_double), intent(in) :: rho(*), sigma(*), lapl(*), tau(*)
    real(c_double), intent(inout) :: exc(*), vrho(*), vsigma(*), vlapl(*), vtau(*)
    integer, intent(out) :: ierr
    integer(c_int) :: cerr
    call c_mgga(int(np, c_int), rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau, cerr)
    ierr = cerr
  end subroutine

end module nxc

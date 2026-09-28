!> Fortran API test: module `nxc` (and, for unpolarized GGA cases, the legacy
!> implicit interface used by SIESTA) against a gen_reference.py case.
!>   test_fortran <case.bin> <model>
program test_fortran
  use, intrinsic :: iso_c_binding, only: c_double, c_int32_t
  use nxc
  implicit none
  real(c_double), parameter :: tol = 1d-10
  character(len=4096) :: casefile, model
  integer(c_int32_t) :: hdr(5)
  integer :: level, ns, np, nsig, ierr, u, nbad
  real(c_double), allocatable :: rho(:,:), sigma(:,:), lapl(:,:), tau(:,:)
  real(c_double), allocatable :: exc_r(:), vrho_r(:,:), vsigma_r(:,:), vtau_r(:,:)
  real(c_double), allocatable :: exc(:), vrho(:,:), vsigma(:,:), vlapl(:,:), vtau(:,:)
  external :: nxc_f90_func_init, nxc_f90_gga_exc_vxc

  call get_command_argument(1, casefile)
  call get_command_argument(2, model)
  open(newunit=u, file=trim(casefile), access='stream', form='unformatted', status='old')
  read(u) hdr
  if (hdr(1) /= 20260928) error stop 'bad magic'
  level = hdr(2); ns = hdr(3); np = hdr(4); nsig = max(hdr(5), 1)
  allocate(rho(np, ns), sigma(np, nsig), lapl(np, ns), tau(np, ns))
  allocate(exc_r(np), vrho_r(np, ns), vsigma_r(np, nsig), vtau_r(np, ns))
  allocate(exc(np), vrho(np, ns), vsigma(np, nsig), vlapl(np, ns), vtau(np, ns))
  sigma = 0; tau = 0; lapl = 0; vsigma_r = 0; vtau_r = 0
  read(u) rho
  if (level >= 1) read(u) sigma
  if (level >= 2) read(u) tau
  read(u) exc_r
  read(u) vrho_r
  if (level >= 1) read(u) vsigma_r
  if (level >= 2) read(u) vtau_r
  close(u)
  write(*, '(a, " (level ", i0, ", nspin ", i0, ", np ", i0, ")")') trim(casefile), level, ns, np

  nbad = 0
  call nxc_init(trim(model), ns, ierr)
  if (ierr /= 0) error stop 'nxc_init failed'
  if (nxc_family() /= merge(NXC_LDA_TYPE, merge(NXC_GGA_TYPE, NXC_MGGA_TYPE, level == 1), level == 0)) then
    write(*, *) ' family BAD', nxc_family()
    nbad = nbad + 1
  end if
  select case (level)
  case (0)
    call nxc_lda(np, rho, exc, vrho, ierr)
  case (1)
    call nxc_gga(np, rho, sigma, exc, vrho, vsigma, ierr)
  case default
    call nxc_mgga(np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau, ierr)
  end select
  if (ierr /= 0) error stop 'evaluation failed'
  call check('module')

  if (level == 1 .and. ns == 1) then        ! legacy interface (unpolarized only)
    exc = 0; vrho = 0; vsigma = 0
    call nxc_f90_func_init(model, len_trim(model), ierr)
    call nxc_f90_gga_exc_vxc(np, rho, sigma, exc, vrho, vsigma, ierr)
    call check('legacy')
  end if

  if (nbad > 0) error stop 1

contains

  real(c_double) function rel(a, r)
    real(c_double), intent(in) :: a(:), r(:)
    rel = maxval(abs(a - r)) / max(maxval(abs(r)), tiny(1d0))
  end function

  subroutine check(what)
    character(len=*), intent(in) :: what
    real(c_double) :: e(4)
    e = 0
    e(1) = rel(exc, exc_r)
    e(2) = rel(reshape(vrho, [np*ns]), reshape(vrho_r, [np*ns]))
    if (level >= 1) e(3) = rel(reshape(vsigma, [np*nsig]), reshape(vsigma_r, [np*nsig]))
    if (level >= 2) e(4) = rel(reshape(vtau, [np*ns]), reshape(vtau_r, [np*ns]))
    if (all(e < tol)) then
      write(*, '(2x, a14, " ok   exc=", es8.1, " vrho=", es8.1, " vsigma=", es8.1, " vtau=", es8.1)') what, e
    else
      write(*, '(2x, a14, " BAD  exc=", es8.1, " vrho=", es8.1, " vsigma=", es8.1, " vtau=", es8.1)') what, e
      nbad = nbad + 1
    end if
  end subroutine

end program test_fortran

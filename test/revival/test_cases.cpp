// C++ API test: evaluate a reference case (gen_reference.py) and compare with pylibnxc.
//
//   test_cases <case.bin> <model> <expect: ok|refuse>
//
// Checks (tolerance 1e-10 on max|diff| / max|ref| per output):
//   1. blocked layout, all points
//   2. interleaved (Libxc) layout
//   3. add mode (outputs pre-filled with 1)
//   4. single-point calls (N = 1) for a few points
//   5. only sub-threshold points -> all zeros
// For expect = refuse, initialization must succeed and evaluation must throw.
#include "nxc.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const double TOL = 1e-10;
int failures = 0;

struct Case {
  int level, ns, np, nsig;
  std::vector<double> rho, sigma, lapl, tau, exc, vrho, vsigma, vtau;
};

Case read_case(const std::string &path){
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  int32_t h[5];
  f.read(reinterpret_cast<char *>(h), sizeof(h));
  if (h[0] != 20260928) throw std::runtime_error("bad magic in " + path);
  Case c{h[1], h[2], h[3], h[4]};
  auto rd = [&](std::vector<double> &v, size_t n){ v.resize(n); f.read(reinterpret_cast<char *>(v.data()), n * 8); };
  size_t nsr = (size_t)c.ns * c.np, nsg = (size_t)c.nsig * c.np;
  rd(c.rho, nsr);
  if (c.level >= 1) rd(c.sigma, nsg);
  if (c.level >= 2) rd(c.tau, nsr);
  rd(c.exc, c.np);
  rd(c.vrho, nsr);
  if (c.level >= 1) rd(c.vsigma, nsg);
  if (c.level >= 2) rd(c.vtau, nsr);
  c.lapl.assign(nsr, 0.0);
  if (!f) throw std::runtime_error("truncated " + path);
  return c;
}

struct Out { std::vector<double> exc, vrho, vsigma, vlapl, vtau; };

Out alloc(const Case &c, double fill){
  return Out{std::vector<double>(c.np, fill), std::vector<double>((size_t)c.ns * c.np, fill),
             std::vector<double>((size_t)std::max(c.nsig, 1) * c.np, fill),
             std::vector<double>((size_t)c.ns * c.np, fill), std::vector<double>((size_t)c.ns * c.np, fill)};
}

void run(nxc_func_type &p, const Case &c, int np, double *rho, double *sigma, double *lapl, double *tau, Out &o){
  if (c.level == 0) nxc_lda_exc_vxc(&p, np, rho, o.exc.data(), o.vrho.data());
  else if (c.level == 1) nxc_gga_exc_vxc(&p, np, rho, sigma, o.exc.data(), o.vrho.data(), o.vsigma.data());
  else nxc_mgga_exc_vxc(&p, np, rho, sigma, lapl, tau, o.exc.data(), o.vrho.data(), o.vsigma.data(),
                        o.vlapl.data(), o.vtau.data());
}

double relerr(const std::vector<double> &a, const std::vector<double> &ref, size_t n, double shift = 0.0){
  double emax = 0.0, rmax = 0.0;
  for (size_t i = 0; i < n; ++i){
    if (!std::isfinite(a[i])) return INFINITY;
    emax = std::max(emax, std::fabs(a[i] - shift - ref[i]));
    rmax = std::max(rmax, std::fabs(ref[i]));
  }
  return rmax > 0 ? emax / rmax : emax;
}

void report(const std::string &what, const Case &c, const Out &o, double shift = 0.0){
  size_t nsr = (size_t)c.ns * c.np, nsg = (size_t)c.nsig * c.np;
  double e[4] = {relerr(o.exc, c.exc, c.np, shift), relerr(o.vrho, c.vrho, nsr, shift),
                 c.level >= 1 ? relerr(o.vsigma, c.vsigma, nsg, shift) : 0.0,
                 c.level >= 2 ? relerr(o.vtau, c.vtau, nsr, shift) : 0.0};
  bool ok = e[0] < TOL && e[1] < TOL && e[2] < TOL && e[3] < TOL;
  std::printf("  %-14s %s  exc=%.1e vrho=%.1e vsigma=%.1e vtau=%.1e\n", what.c_str(), ok ? "ok " : "BAD",
              e[0], e[1], e[2], e[3]);
  if (!ok) ++failures;
}

// blocked (nblk, np) -> interleaved (np, nblk)
std::vector<double> interleave(const std::vector<double> &a, int nblk, int np){
  if (a.empty()) return a;
  std::vector<double> b(a.size());
  for (int k = 0; k < nblk; ++k) for (int i = 0; i < np; ++i) b[(size_t)i * nblk + k] = a[(size_t)k * np + i];
  return b;
}
std::vector<double> deinterleave(const std::vector<double> &b, int nblk, int np){
  if (b.empty()) return b;
  std::vector<double> a(b.size());
  for (int k = 0; k < nblk; ++k) for (int i = 0; i < np; ++i) a[(size_t)k * np + i] = b[(size_t)i * nblk + k];
  return a;
}
// point subset (blocked layout)
std::vector<double> pick(const std::vector<double> &a, int nblk, int np, const std::vector<int> &idx){
  if (a.empty()) return a;
  std::vector<double> b((size_t)nblk * idx.size());
  for (int k = 0; k < nblk; ++k) for (size_t j = 0; j < idx.size(); ++j) b[k * idx.size() + j] = a[(size_t)k * np + idx[j]];
  return b;
}
} // namespace

int run_case(int argc, char **argv);

int main(int argc, char **argv){
  try{
    return run_case(argc, argv);
  }catch (const std::exception &e){
    std::printf("  EXCEPTION      BAD  %s\n", e.what());
    return 1;
  }
}

int run_case(int argc, char **argv){
  if (argc < 4){ std::cerr << "usage: test_cases case.bin model ok|refuse\n"; return 2; }
  Case c = read_case(argv[1]);
  std::string model = argv[2], expect = argv[3];
  std::printf("%s  (level %d, nspin %d, np %d)\n", argv[1], c.level, c.ns, c.np);
  int nsig = std::max(c.nsig, 1);

  nxc_func_type p;
  func_param fp;
  if (nxc_func_init(&p, model, fp, c.ns) != 0){ std::printf("  init FAILED\n"); return 1; }

  if (expect == "refuse"){
    Out o = alloc(c, 0.0);
    try{
      run(p, c, c.np, c.rho.data(), c.sigma.data(), c.lapl.data(), c.tau.data(), o);
      std::printf("  refuse         BAD (evaluation did not fail)\n");
      return 1;
    }catch (const std::exception &e){
      std::printf("  refuse         ok  (%s)\n", e.what());
      return 0;
    }
  }

  // 1. blocked layout
  Out o = alloc(c, 0.0);
  run(p, c, c.np, c.rho.data(), c.sigma.data(), c.lapl.data(), c.tau.data(), o);
  report("blocked", c, o);

  // 2. interleaved layout
  {
    nxc_func_type pi; func_param fpi; fpi.interleaved = 1;
    nxc_func_init(&pi, model, fpi, c.ns);
    auto r = interleave(c.rho, c.ns, c.np), s = interleave(c.sigma, nsig, c.np);
    auto l = interleave(c.lapl, c.ns, c.np), t = interleave(c.tau, c.ns, c.np);
    Out oi = alloc(c, 0.0);
    run(pi, c, c.np, r.data(), s.data(), l.data(), t.data(), oi);
    oi.vrho = deinterleave(oi.vrho, c.ns, c.np);
    if (c.level >= 1) oi.vsigma = deinterleave(oi.vsigma, nsig, c.np);
    if (c.level >= 2) oi.vtau = deinterleave(oi.vtau, c.ns, c.np);
    report("interleaved", c, oi);
  }

  // 3. add mode
  {
    nxc_func_type pa; func_param fpa; fpa.add = 1;
    nxc_func_init(&pa, model, fpa, c.ns);
    Out oa = alloc(c, 1.0);
    run(pa, c, c.np, c.rho.data(), c.sigma.data(), c.lapl.data(), c.tau.data(), oa);
    report("add", c, oa, 1.0);
  }

  // 4. single points (N = 1), a handful spread over the case
  {
    int bad = 0;
    double worst = 0.0;
    for (int j = 0; j < 8; ++j){
      int i = (int)((long long)j * (c.np - 1) / 7);
      std::vector<int> idx{i};
      Case c1 = c; c1.np = 1;
      c1.rho = pick(c.rho, c.ns, c.np, idx); c1.sigma = pick(c.sigma, nsig, c.np, idx);
      c1.lapl = pick(c.lapl, c.ns, c.np, idx); c1.tau = pick(c.tau, c.ns, c.np, idx);
      c1.exc = pick(c.exc, 1, c.np, idx); c1.vrho = pick(c.vrho, c.ns, c.np, idx);
      c1.vsigma = c.level >= 1 ? pick(c.vsigma, nsig, c.np, idx) : std::vector<double>();
      c1.vtau = c.level >= 2 ? pick(c.vtau, c.ns, c.np, idx) : std::vector<double>();
      Out o1 = alloc(c1, 0.0);
      run(p, c1, 1, c1.rho.data(), c1.sigma.data(), c1.lapl.data(), c1.tau.data(), o1);
      // compare against the full-batch result at the same point (absolute scale of the case)
      auto cmp = [&](const std::vector<double> &a, const std::vector<double> &full, int nblk){
        double sc = 0; for (double v : full) sc = std::max(sc, std::fabs(v));
        for (int k = 0; k < nblk; ++k){
          double d = std::fabs(a[k] - full[(size_t)k * c.np + i]) / (sc > 0 ? sc : 1);
          worst = std::max(worst, d); if (!(d < TOL)) ++bad;
        }
      };
      cmp(o1.exc, o.exc, 1); cmp(o1.vrho, o.vrho, c.ns);
      if (c.level >= 1) cmp(o1.vsigma, o.vsigma, nsig);
      if (c.level >= 2) cmp(o1.vtau, o.vtau, c.ns);
    }
    std::printf("  %-14s %s  worst=%.1e\n", "single-point", bad ? "BAD" : "ok ", worst);
    if (bad) ++failures;
  }

  // 5. all points below the threshold
  {
    Case cz = c;
    for (auto &v : cz.rho) v *= 1e-14;
    for (auto &v : cz.sigma) v *= 1e-28;
    for (auto &v : cz.tau) v *= 1e-24;
    Out oz = alloc(c, 7.0);
    run(p, cz, c.np, cz.rho.data(), cz.sigma.data(), cz.lapl.data(), cz.tau.data(), oz);
    bool zero = true;
    for (double v : oz.exc) zero &= (v == 0.0);
    for (double v : oz.vrho) zero &= (v == 0.0);
    std::printf("  %-14s %s\n", "below-thresh", zero ? "ok " : "BAD");
    if (!zero) ++failures;
  }

  return failures ? 1 : 0;
}

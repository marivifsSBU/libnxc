#include "nxc.h"
#include "nxc_func.h"
#include <torch/script.h> // One-stop header.
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <unordered_map>

#define STRINGIZER(arg)     #arg
#define STR_VALUE(arg)      STRINGIZER(arg)
#ifdef NXC_DEFAULTPATH_STR
const std::string default_modeldir = NXC_DEFAULTPATH_STR;
#else
#define NXC_DEFAULTPATH_STRING STR_VALUE(NXC_DEFAULTPATH)
const std::string default_modeldir = NXC_DEFAULTPATH_STRING;
#endif

namespace {

std::string join_path(const std::string &dir, const std::string &name){
  if (dir.empty()) return name;
  return (std::filesystem::path(dir) / name).string();
}

// Leaf tensor over a host array of nblk*np doubles; returns the (nblk, np) view in
// spin/component-blocked order.  Blocked (default): a[ib*np + ip] (Fortran a(np, nblk)).
// Interleaved (Libxc): a[ip*nblk + ib].
torch::Tensor host_leaf(double *a, int nblk, int np, bool interleaved, torch::Tensor &leaf){
  leaf = torch::from_blob(a, {(int64_t)nblk * np}, options_dp.requires_grad(true));
  if (interleaved) return leaf.view({np, nblk}).t();
  return leaf.view({nblk, np});
}

// Copy (or add) a gradient/energy tensor with the same layout as the host array.
void to_host(const torch::Tensor &src, double *dst, int64_t n, bool add){
  torch::Tensor s = src.defined() ? src.to(torch::kCPU).contiguous() : torch::zeros({n}, options_dp);
  const double *p = s.data_ptr<double>();
  if (add){
    for (int64_t i = 0; i < n; ++i) dst[i] += p[i];
  }else{
    for (int64_t i = 0; i < n; ++i) dst[i] = p[i];
  }
}

void zero_host(double *dst, int64_t n, bool add){
  if (!add && dst != nullptr)
    for (int64_t i = 0; i < n; ++i) dst[i] = 0.0;
}

torch::Tensor nonneg(const torch::Tensor &x){
  // '>=': gradient 1 at x == 0 (one-sided derivative for an empty spin channel)
  return torch::where(x >= 0, x, torch::zeros_like(x));
}

torch::Tensor floored(const torch::Tensor &x, double floor){
  // value max(x, floor) with the gradient passed through, i.e. the potential is the
  // derivative evaluated at the floored point, as Libxc does with its density and sigma
  // thresholds.  Protects models with rho^(1/3) or sqrt(sigma) terms from 0*inf at an
  // empty spin channel or sigma = 0.  floor <= 0: identity.
  if (floor <= 0) return x;
  return x + (torch::clamp_min(x, floor) - x).detach();
}


} // namespace

std::string resolve_model_dir(const std::string &modelname){
  std::vector<std::string> candidates = {modelname};
  if (const char *env = std::getenv("NXC_MODELPATH"))
    candidates.push_back(join_path(env, modelname));
  candidates.push_back(join_path(default_modeldir, modelname));
  for (const auto &c : candidates)
    if (std::filesystem::exists(join_path(c, "xc"))) return c;
  return "";
}

GridFunc::GridFunc(std::string modelname){
  std::string dir = resolve_model_dir(modelname);
  if (dir.empty())
    throw std::runtime_error("libnxc: model '" + modelname + "' not found (looked for <name>/xc as "
                             "given, under $NXC_MODELPATH and under " + default_modeldir + ")");
  try {
    // models written by the libnxc_revival exporter embed a manifest; those declaring
    // "thresholds": "internal" handle empty spin channels and sigma = 0 themselves
    std::unordered_map<std::string, std::string> extra{{"manifest.json", ""}};
    model.energy = torch::jit::load(join_path(dir, "xc"), std::nullopt, extra);
    internal_thresholds = extra["manifest.json"].find("\"thresholds\": \"internal\"") != std::string::npos;
  }catch (const c10::Error& e) {
    throw std::runtime_error("libnxc: error loading the model " + join_path(dir, "xc") + ": " +
                             e.what_without_backtrace());
  }
  model.energy.eval();
}

void GridFunc::init(func_param fp, int nspin){
  this->nspin = nspin;
  add = fp.add;
  edens = fp.edens;
  gamma = fp.gamma;
  interleaved = fp.interleaved;
  rho_threshold = fp.rho_threshold;
  sigma_threshold = internal_thresholds ? 0.0 : fp.sigma_threshold;
  spin_threshold = internal_thresholds ? 0.0 : fp.spin_threshold;
  tau_threshold = internal_thresholds ? 0.0 : fp.tau_threshold;
  tau_w_cap = internal_thresholds ? false : (fp.tau_w_cap != 0);
  if (!edens){
    tcell = torch::from_blob(fp.cell, 9, options_dp).view({3,3});
    tgrid = torch::from_blob(fp.grid, 3, options_int);
    torch::Tensor U = torch::zeros({3,3}, options_dp);
    for(int i = 0; i< 3; ++i){
      U.select(1,i) = tcell.select(1,i)/tgrid.select(0,i);
    }
    V_cell = torch::abs(torch::det(U));
  }
  if (fp.cuda) this->to_cuda();
}

void GridFunc::to_cuda(){
  model.energy.to(at::kCUDA);
  device = DEVICE_CUDA;
  std::cout << "libnxc:Using CUDA" << std::endl;
}

/* Shared evaluation for LDA (level 0), GGA (1) and meta-GGA (2).
 *
 * Host arrays (np grid points; nspin = 1 or 2):
 *   rho   nspin*np
 *   sigma 1|3 contracted gradients (aa, ab, bb), or 3|6 gradient components if gamma
 *   lapl, tau  nspin*np
 * The model receives (N, 2|5|9) columns rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
 * lapl_a, lapl_b, tau_a, tau_b for the points with rho_a + rho_b > rho_threshold and
 * returns eps_xc (energy per particle).  Negative rho/sigma_aa/sigma_bb/tau are clipped to
 * 0; spin densities and sigma_aa, sigma_bb are floored at spin_threshold / sigma_threshold
 * with the gradient passed through, as in Libxc; for meta-GGAs sigma_ss is capped at
 * 8 rho_s tau_s (Libxc's Fermi-hole curvature enforcement).  All skipped for models whose
 * embedded manifest declares "thresholds": "internal".  Potentials are the gradients of
 * E = sum_p eps_p (rho_a + rho_b)_p w.r.t. the host arrays (same layout as the inputs);
 * points below the threshold get exc = v = 0.
 */
void GridFunc::evaluate(int level, int np, double rho[], double sigma[], double lapl[], double tau[],
                        double *exc, double vrho[], double vsigma[], double vlapl[], double vtau[]){
  const bool pol = (nspin == NXC_POLARIZED);
  const int ns = pol ? 2 : 1;
  const int nsig = gamma ? (pol ? 6 : 3) : (pol ? 3 : 1);
  const int64_t npe = edens ? np : 1;

  torch::Tensor lr, ls, ll, lt;                      // leaves over the host arrays
  torch::Tensor r = host_leaf(rho, ns, np, interleaved, lr);
  torch::Tensor rho_ab = pol ? r : r.expand({2, np}) * 0.5;

  std::vector<torch::Tensor> cols = {floored(nonneg(rho_ab.select(0, 0)), spin_threshold),
                                     floored(nonneg(rho_ab.select(0, 1)), spin_threshold)};
  if (level >= 1){
    torch::Tensor s = host_leaf(sigma, nsig, np, interleaved, ls);
    torch::Tensor saa, sab, sbb;
    if (gamma){        // gradient components (x, y, z)_a [, (x, y, z)_b]
      torch::Tensor ga = pol ? s.slice(0, 0, 3) : s * 0.5;
      torch::Tensor gb = pol ? s.slice(0, 3, 6) : s * 0.5;
      saa = (ga * ga).sum(0);
      sab = (ga * gb).sum(0);
      sbb = (gb * gb).sum(0);
    }else if (pol){
      saa = s.select(0, 0); sab = s.select(0, 1); sbb = s.select(0, 2);
    }else{
      saa = sab = sbb = s.select(0, 0) * 0.25;
    }
    cols.push_back(floored(nonneg(saa), sigma_threshold));
    cols.push_back(sab);                       // may be negative for polarized densities
    cols.push_back(floored(nonneg(sbb), sigma_threshold));
  }
  if (level >= 2){
    torch::Tensor l = host_leaf(lapl, ns, np, interleaved, ll);
    torch::Tensor t = host_leaf(tau, ns, np, interleaved, lt);
    torch::Tensor l_ab = pol ? l : l.expand({2, np}) * 0.5;
    torch::Tensor t_ab = pol ? t : t.expand({2, np}) * 0.5;
    cols.push_back(l_ab.select(0, 0));         // laplacian may be negative: not clipped
    cols.push_back(l_ab.select(0, 1));
    cols.push_back(floored(nonneg(t_ab.select(0, 0)), tau_threshold));
    cols.push_back(floored(nonneg(t_ab.select(0, 1)), tau_threshold));
    if (tau_w_cap){
      // Libxc (>= 5) enforces the Fermi-hole curvature tau_s >= tau_W,s for meta-GGAs by
      // capping sigma_ss <= 8 rho_s tau_s, with derivatives evaluated at the capped point.
      // Matters with nonlinear core corrections (core density without core tau).
      for (int s2 = 0; s2 < 2; ++s2){
        torch::Tensor &sg = cols[2 + 2 * s2];
        torch::Tensor cap = 8.0 * cols[s2] * cols[7 + s2];
        sg = sg + (torch::minimum(sg, cap) - sg).detach();
      }
      // as Libxc: keep sigma_ab within +-(sigma_aa + sigma_bb)/2 after the cap (for
      // unpolarized input this keeps sigma_aa = sigma_ab = sigma_bb)
      torch::Tensor s_ave = 0.5 * (cols[2] + cols[4]).detach();
      cols[3] = cols[3] + (torch::maximum(torch::minimum(cols[3], s_ave), -s_ave) - cols[3]).detach();
    }
  }

  torch::Tensor rho_sum = rho_ab.sum(0);                       // raw (as in pylibnxc)
  torch::Tensor filter = (cols[0] + cols[1]).detach() > rho_threshold;
  torch::Tensor idx = torch::nonzero(filter).reshape({-1});
  const int64_t nf = idx.size(0);

  if (nf == 0){
    zero_host(exc, npe, add);
    zero_host(vrho, (int64_t)ns * np, add);
    if (level >= 1) zero_host(vsigma, (int64_t)nsig * np, add);
    if (level >= 2){ zero_host(vlapl, (int64_t)ns * np, add); zero_host(vtau, (int64_t)ns * np, add); }
    return;
  }

  torch::Tensor inp = torch::stack(cols, 1).index_select(0, idx);   // (nf, ncol)
  if (nf == 1) inp = inp.repeat({2, 1});   // models that squeeze() fail on a single point
  if (device == DEVICE_CUDA) inp = inp.to(at::kCUDA);

  torch::Tensor texc = model.energy.forward({inp}).toTensor().to(torch::kCPU).reshape({-1});
  if (texc.size(0) != inp.size(0))
    throw std::runtime_error("libnxc: model returned " + std::to_string(texc.size(0)) +
                             " values for " + std::to_string(inp.size(0)) + " points");
  texc = texc.slice(0, 0, nf);
  if (!torch::isfinite(texc).all().item<bool>()){
    std::ostringstream msg;
    torch::Tensor badidx = torch::nonzero(~torch::isfinite(texc)).reshape({-1});
    msg << "libnxc: non-finite exc at " << badidx.size(0) << " of " << nf << " points; first inputs "
        << "(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb[, lapl_a, lapl_b, tau_a, tau_b]):";
    torch::Tensor rows = inp.index_select(0, badidx.slice(0, 0, 3)).to(torch::kCPU).detach();
    msg.precision(6);
    for (int64_t i = 0; i < rows.size(0); ++i){
      msg << "\n   ";
      for (int64_t j = 0; j < rows.size(1); ++j) msg << " " << std::scientific << rows[i][j].item<double>();
    }
    throw std::runtime_error(msg.str());
  }
  torch::Tensor texc_full = torch::zeros({np}, options_dp).index_put({idx}, texc);
  torch::Tensor Exc = torch::dot(texc_full, rho_sum);
  Exc.backward();

  for (auto *leaf : {&lr, &ls, &ll, &lt}){
    if (leaf->defined() && leaf->grad().defined() && !torch::isfinite(leaf->grad()).all().item<bool>())
      throw std::runtime_error("libnxc: non-finite potential");
  }

  if (edens) to_host(texc_full.detach(), exc, np, add);
  else       to_host(Exc.detach().reshape({1}), exc, 1, add);
  to_host(lr.grad(), vrho, (int64_t)ns * np, add);
  if (level >= 1) to_host(ls.grad(), vsigma, (int64_t)nsig * np, add);
  if (level >= 2){
    to_host(ll.grad(), vlapl, (int64_t)ns * np, add);
    to_host(lt.grad(), vtau, (int64_t)ns * np, add);
  }
}

void LDAFunc::exc_vxc(int np, double rho[], double * exc, double vrho[]){
  evaluate(0, np, rho, nullptr, nullptr, nullptr, exc, vrho, nullptr, nullptr, nullptr);
}

void GGAFunc::exc_vxc(int np, double rho[], double sigma[],
        double * exc, double vrho[], double vsigma[]){
  evaluate(1, np, rho, sigma, nullptr, nullptr, exc, vrho, vsigma, nullptr, nullptr);
}

void MGGAFunc::exc_vxc(int np, double rho[], double sigma[], double lapl[],
        double tau[], double * exc, double vrho[], double vsigma[], double vlapl[], double vtau[]){
  evaluate(2, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau);
}

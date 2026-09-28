#include "nxc_func.h"
#include <filesystem>
#include <stdexcept>

namespace {
// LDA/GGA/MGGA from the first token of the model directory name (as in pylibnxc)
int family_from_name(const std::string &model){
  std::filesystem::path p(model);
  std::string base = p.filename().string();
  if (base.empty()) base = p.parent_path().filename().string();   // trailing '/'
  std::string fam = base.substr(0, base.find('_'));
  if (fam == "LDA") return LDA_TYPE;
  if (fam == "GGA") return GGA_TYPE;
  if (fam == "MGGA") return MGGA_TYPE;
  return -1;
}
}

std::shared_ptr<NXCFunc> get_functional(std::string model)
{
  int family = -1;
  auto it = xctypes.find(model);
  if (it != xctypes.end()){
    family = it->second;                       // shipped model names
  }else if (!resolve_model_dir(model).empty()){
    family = family_from_name(model);          // custom grid model: <dir>/xc
    if (family < 0)
      throw std::runtime_error("libnxc: cannot infer LDA/GGA/MGGA from model name '" + model +
                               "' (directory name must start with LDA_, GGA_ or MGGA_)");
  }
  switch(family){
    case LDA_TYPE:  return std::make_shared<LDAFunc>(model);
    case GGA_TYPE:  return std::make_shared<GGAFunc>(model);
    case MGGA_TYPE: return std::make_shared<MGGAFunc>(model);
  }
  return std::make_shared<AtomicFunc>(model);  // NeuralXC (xc_X, basis_X, projector_X files)
}

from pyscf import dft
from ..adapters import PySCFNXC, NXCAdapter, get_nxc_adapter
from ..functional import AtomicFunc, GridFunc, HMFunc, LibNXCFunctional
from .utils import parse_xc_code, find_max_level, xc_family
import numpy as np
import os
from functools import partial


def KS(mol, method, nxc='', nxc_kind='grid', **kwargs):
    """ Wrapper for the pyscf RKS and UKS class
    that uses a libnxc functionals
    """
    mf = method(mol, **kwargs)
    if nxc != '':
        if nxc_kind.lower() == 'atomic':
            model = get_nxc_adapter('pyscf', nxc)
            mf.get_veff = veff_mod_atomic(mf, model)
        elif nxc_kind.lower() == 'grid':
            parsed_xc = parse_xc_code(nxc)
            hyb = parsed_xc[0][0]
            xctype = find_max_level(parsed_xc)
            # PySCF >= 2 still parses mf.xc with Libxc in a few places
            # (is_hybrid_xc, do_nlc, dump_flags): use a valid placeholder with
            # the same amount of exact exchange. The evaluation itself goes
            # through ni.eval_xc, which ignores the xc_code string.
            mf.xc = '{:.12g}*HF + {:.12g}*LDA,'.format(hyb, 1 - hyb) if hyb else 'LDA,'
            dft.libxc.define_xc_(mf._numint,
                                 make_eval_xc(parsed_xc[1], xctype),
                                 xctype,
                                 hyb=hyb)
            mf.nxc = nxc
        else:
            raise ValueError(
                "{} not a valid nxc_kind. Valid options are 'atomic' or 'grid'"
                .format(nxc_kind))
    return mf


RKS = partial(KS, method=dft.RKS)
UKS = partial(KS, method=dft.UKS)

_model_cache = {}


def _load_grid_model(name):
    if name not in _model_cache:
        _model_cache[name] = LibNXCFunctional(name, kind='grid')
    return _model_cache[name]


def _unpack_rho(rho, spin, xctype):
    """ PySCF density array -> pylibnxc input dict (Libxc variables).
    PySCF >= 2 passes meta-GGA densities as (rho, dx, dy, dz, tau); a
    (…, 6, N) array with the laplacian in row 4 is accepted as well.
    """
    if spin == 0:
        rho = np.asarray(rho)
        if rho.ndim == 1:
            rho = rho.reshape(1, -1)
        inp = {'rho': np.ascontiguousarray(rho[0])}
        if xctype in ('GGA', 'MGGA'):
            inp['sigma'] = np.einsum('xg,xg->g', rho[1:4], rho[1:4])
        if xctype == 'MGGA':
            inp['tau'] = np.ascontiguousarray(rho[5] if len(rho) == 6 else rho[4])
    else:
        rho_a, rho_b = (np.asarray(r) for r in rho)
        if rho_a.ndim == 1:
            rho_a = rho_a.reshape(1, -1)
            rho_b = rho_b.reshape(1, -1)
        inp = {'rho': np.stack([rho_a[0], rho_b[0]])}
        if xctype in ('GGA', 'MGGA'):
            inp['sigma'] = np.stack([
                np.einsum('xg,xg->g', rho_a[1:4], rho_a[1:4]),
                np.einsum('xg,xg->g', rho_a[1:4], rho_b[1:4]),
                np.einsum('xg,xg->g', rho_b[1:4], rho_b[1:4])])
        if xctype == 'MGGA':
            it = 5 if len(rho_a) == 6 else 4
            inp['tau'] = np.stack([rho_a[it], rho_b[it]])
    return inp


def make_eval_xc(funcs, xctype):
    """ PySCF >= 2 custom eval_xc for a linear combination [(name, factor), ...]
    of grid-based libnxc models (not atomic). See pyscf documentation of eval_xc.
    """
    for name, _ in funcs:
        _load_grid_model(name)

    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
                verbose=None, **kwargs):
        if deriv > 1:
            raise NotImplementedError(
                'libnxc grid functionals provide exc and vxc only (deriv <= 1)')
        inp = _unpack_rho(rho, spin, xctype)
        ng = inp['rho'].shape[-1]
        exc = np.zeros(ng)
        vrho = np.zeros((ng, 2)) if spin else np.zeros(ng)
        vsigma = vtau = None
        if xctype != 'LDA':
            vsigma = np.zeros((ng, 3)) if spin else np.zeros(ng)
        if xctype == 'MGGA':
            vtau = np.zeros((ng, 2)) if spin else np.zeros(ng)
        for name, factor in funcs:
            level = {'LDA': 0, 'GGA': 1, 'MGGA': 2}[xc_family(name)]
            keys = ['rho', 'sigma', 'tau'][:level + 1]
            output = _load_grid_model(name).compute({k: inp[k] for k in keys},
                                                    do_vxc=deriv > 0)
            exc += factor * output['zk']
            if deriv > 0:
                vrho += factor * output['vrho']
                if 'vsigma' in output:
                    vsigma += factor * output['vsigma']
                if 'vtau' in output:
                    vtau += factor * output['vtau']
        # no vlapl: PySCF >= 2 meta-GGAs do not depend on the laplacian
        return exc, (vrho, vsigma, None, vtau), None, None

    return eval_xc


def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
            verbose=None):
    """ Evaluate a libnxc xc_code string directly (PySCF eval_xc signature) """
    parsed_xc = parse_xc_code(xc_code)
    return make_eval_xc(parsed_xc[1], find_max_level(parsed_xc))(
        xc_code, rho, spin, relativity, deriv, omega, verbose)


def veff_mod_atomic(mf, model):
    """ Wrapper to get the modified get_veff() that uses a NeuralXC
    atomic potential
    """
    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
                verbose=None):
        rho0 = rho[:1]
        gamma = None

        exc, V_nxc = model.compute(rho0.flatten(), edens=True)

        vrho = V_nxc

        vgamma = np.zeros_like(V_nxc)
        vlapl = None
        vtau = None
        vxc = (vrho, vgamma, vlapl, vtau)
        fxc = None  # 2nd order functional derivative
        kxc = None  # 3rd order functional derivative

        return exc, vxc, fxc, kxc

    def get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        mf.define_xc_(mf.xc, 'GGA')  #TODO: This doesn't seem quite right
        veff = dft.rks.get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
        mf.define_xc_(eval_xc, 'GGA')
        model.initialize(mf.grids.coords, mf.grids.weights, mol)
        vnxc = dft.rks.get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
        veff[:, :] += (vnxc[:, :] - vnxc.vj[:, :])
        veff.exc += vnxc.exc
        return veff

    return get_veff

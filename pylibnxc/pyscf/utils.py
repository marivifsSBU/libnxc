import os
import re

_XC_LEVELS = {'LDA': 0, 'GGA': 1, 'MGGA': 2}


def parse_xc_code(xc_code):
    """ Parse pyscf style xc_code, that can contain +,*,- and combinations
    of functionals, e.g. '0.25*HF + 0.75*GGA_X_PBE, GGA_C_PBE'.

    Returns ((hyb, 0, 0), [(name, factor), ...]).  Names are kept as given; they
    may be paths to custom models (paths must exist and must not contain ',' or '*';
    '+'/'-' inside an existing path, e.g. .../XC-DFT/..., are handled).
    Libxc's parser is not used: PySCF >= 2 rejects names it does not know.
    """
    hyb = 0.0
    funcs = []
    for part in xc_code.split(','):
        part = part.strip()
        if os.path.exists(part):             # a bare path, possibly containing '-'
            funcs.append((part, 1.0))
            continue
        part = part.replace(' ', '')
        # split before '+'/'-' unless it is an exponent sign (e.g. 1e-3)
        toks = [t for t in re.split(r'(?<!\d[eE])(?=[+-])', part) if t]
        # re-join pieces of a path that contains '+' or '-' (e.g. .../XC-DFT/...)
        merged = []
        for tok in toks:
            if merged and '/' in merged[-1] and not os.path.exists(merged[-1].split('*')[-1]) \
                    and os.path.exists((merged[-1] + tok).split('*')[-1]):
                merged[-1] += tok
            elif merged and '/' in merged[-1] and not os.path.exists(merged[-1].split('*')[-1]) \
                    and '/' in tok and not re.match(r'[+-]?[\d.]', tok):
                merged[-1] += tok
            else:
                merged.append(tok)
        for tok in merged:
            sign, tok = (tok[0], tok[1:]) if tok[0] in '+-' else ('+', tok)
            fac = -1.0 if sign == '-' else 1.0
            name = None
            for f in tok.split('*'):
                try:
                    fac *= float(f)
                except ValueError:
                    name = f
            if name is None:
                raise ValueError('No functional name in term {}'.format(tok))
            if name.upper() == 'HF':
                hyb += fac
            else:
                funcs.append((name, fac))
    return (hyb, 0, 0), funcs


def xc_family(name):
    """ Family (LDA/GGA/MGGA) from the first token of the model name/directory """
    fam = os.path.basename(os.path.normpath(name)).split('_')[0].upper()
    if fam not in _XC_LEVELS:
        raise ValueError(
            'Cannot infer LDA/GGA/MGGA family from model name {}'.format(name))
    return fam


def find_max_level(parsed_xc):
    levels = [_XC_LEVELS[xc_family(name)] for name, _ in parsed_xc[1]]
    return {v: k for k, v in _XC_LEVELS.items()}[max(levels)]

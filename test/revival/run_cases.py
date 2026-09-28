"""Run a test executable over all reference cases (used by CTest).

    python run_cases.py <cases.txt> <exe> cxx|fortran
"""
import subprocess
import sys

cases_file, exe, kind = sys.argv[1:4]
casedir = cases_file.rsplit('/', 1)[0]
failed = []
for line in open(cases_file):
    if not line.strip():
        continue
    case, model, level, nspin, expect = line.split()
    if kind == 'fortran' and expect != 'ok':
        continue                    # refusal is checked through the C++ API
    args = [exe, f'{casedir}/{case}.bin', model] + ([expect] if kind == 'cxx' else [])
    r = subprocess.run(args, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stdout.write(r.stderr)
        failed.append(case)
print(f'\n{kind}: {len(failed)} failed' + (': ' + ', '.join(failed) if failed else ''))
sys.exit(1 if failed else 0)

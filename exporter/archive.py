"""Make exported model directories self-contained and free of local paths.

- `bundle_sources`: copy the equinox checkpoints the model was exported from into
  `<model>/source/` and record them in the manifest by relative path (with sha256), so the
  numerical gate can be re-run from the repository alone.
- `strip_archive`: rewrite a TorchScript archive without its `*.debug_pkl` entries (they only
  hold source-range information, including absolute file names of the exporting machine) and
  with an updated embedded manifest. TorchScript loads such archives normally.

    python -m exporter.archive models/GGA_XC_MYCLONE      # sanitize an existing export
"""
import hashlib
import json
import os
import shutil
import sys
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def _files(entry):
    """Files that make up one manifest source entry (fx/fc prefix or one MXC checkpoint)."""
    if 'path' in entry:
        return [entry['path'], entry['path'] + '.json'], 'path'
    return [entry['prefix'] + '.eqx', entry['prefix'] + '.json'], 'prefix'


def bundle_sources(mdir, manifest):
    """Copy the source checkpoints into <mdir>/source and make the manifest paths relative.
    Already relative paths (resolved against mdir) are left as they are."""
    sdir = os.path.join(mdir, 'source')
    for key, entry in manifest['sources'].items():
        files, field = _files(entry)
        if not os.path.isabs(entry[field]):
            continue
        os.makedirs(sdir, exist_ok=True)
        for f in files:
            shutil.copy2(f, os.path.join(sdir, os.path.basename(f)))
        checked = files[0]
        if sha256(os.path.join(sdir, os.path.basename(checked))) != entry['sha256']:
            raise RuntimeError(f'{key}: copied checkpoint does not match the recorded sha256')
        entry[field] = os.path.join('source', os.path.basename(entry[field]))
    return manifest


def resolve_sources(sources, mdir):
    """Manifest 'sources' with paths made absolute relative to the model directory."""
    out = {}
    for key, entry in sources.items():
        entry = dict(entry)
        field = 'path' if 'path' in entry else 'prefix'
        if not os.path.isabs(entry[field]):
            entry[field] = os.path.join(mdir, entry[field])
        out[key] = entry
    return out


def strip_archive(xc_path, manifest=None):
    """Rewrite the TorchScript archive without debug pickles (and with a new manifest)."""
    tmp = xc_path + '.tmp'
    with zipfile.ZipFile(xc_path) as zin, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as zout:
        for info in zin.infolist():
            if info.filename.endswith('.debug_pkl'):
                continue
            data = zin.read(info.filename)
            if manifest is not None and info.filename.endswith('/extra/manifest.json'):
                data = json.dumps(manifest).encode()
            zi = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
            zout.writestr(zi, data)
    os.replace(tmp, xc_path)


def sanitize(mdir):
    mdir = os.path.abspath(mdir)
    mpath = os.path.join(mdir, 'manifest.json')
    manifest = json.load(open(mpath))
    if 'sources' in manifest:
        bundle_sources(mdir, manifest)
    with open(mpath, 'w') as f:
        json.dump(manifest, f, indent=1)
    strip_archive(os.path.join(mdir, 'xc'), manifest)
    cpath = os.path.join(mdir, 'check.json')
    if os.path.exists(cpath):
        check = json.load(open(cpath))
        check['model_dir'] = os.path.basename(mdir)
        with open(cpath, 'w') as f:
            json.dump(check, f, indent=1)


if __name__ == '__main__':
    for d in sys.argv[1:]:
        sanitize(d)
        print('sanitized', d)

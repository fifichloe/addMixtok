"""Configured training cache paths and file utilities."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
FRESH=Path(os.environ['MIXTOK_BASE_FEATURE_CACHE'])
def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')

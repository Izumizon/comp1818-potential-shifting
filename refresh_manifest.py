"""Record the current delivery files; rerun only after deliberate delivery edits.

This does not validate scientific evidence or bless unreviewed changes. Use the
existing preservation checks to retain immutable experimental provenance.
"""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    exclude = {'.git', '.venv', '__pycache__', '.pytest_cache'}
    files = sorted(p for p in root.rglob('*') if p.is_file() and
        not any(part in exclude for part in p.relative_to(root).parts) and
        p.suffix not in {'.pyc', '.nbc', '.nbi'} and p.name != 'SHA256SUMS.json')
    records = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root/'SHA256SUMS.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Recorded {len(records)} delivery files.')


if __name__ == '__main__':
    main()

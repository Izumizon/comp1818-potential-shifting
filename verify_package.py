"""Verify the delivered file manifest using only Python's standard library.

Run from any directory: python path/to/verify_package.py
Extra files such as freshly generated caches are allowed, but every recorded
delivered file must exist and retain its recorded SHA-256 digest.
"""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    records = json.loads((root / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    failed = []
    for relative, digest in records.items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failed.append(relative)
    if failed:
        raise SystemExit('Missing or modified delivered files: ' + ', '.join(failed))
    print(f'PASS: {len(records)} delivered files match SHA-256 manifest.')


if __name__ == '__main__':
    main()

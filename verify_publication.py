"""Verify public GitHub artefacts with unauthenticated standard-library GETs.

No GitHub CLI credentials, Authorization header, cookie jar or owner session is
used. A successful HTML status alone is insufficient: selected source/dataset
bytes must also match the local delivery. Optional release-archive verification
downloads and checks every member against its embedded manifest.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
REPOSITORY = 'Izumizon/comp1818-potential-shifting'
BASE = 'https://github.com/' + REPOSITORY
RAW = 'https://raw.githubusercontent.com/' + REPOSITORY + '/main/'


def get(url):
    # Fresh opener for each request; there is deliberately no CookieProcessor
    # or HTTP authentication handler, and credentials are never read from gh.
    opener = urllib.request.build_opener()
    headers = {'User-Agent': 'COMP1818-public-delivery-verification', 'Accept': '*/*'}
    request = urllib.request.Request(url, headers=headers)
    with opener.open(request, timeout=60) as response:
        assert response.status == 200, (url, response.status)
        assert '/login' not in response.url, 'redirected to authentication'
        body = response.read()
        result = {'url': url, 'status': response.status, 'bytes': len(body),
                  'sha256': hashlib.sha256(body).hexdigest(),
                  'content_type': response.headers.get('Content-Type')}
    return body, result


def main(archive_url=None, archive_path=None, output='verification/public_access_r9.json'):
    checks = []
    payload, record = get('https://api.github.com/repos/' + REPOSITORY)
    meta = json.loads(payload)
    assert meta['private'] is False and meta['full_name'] == REPOSITORY
    checks.append(record)
    for url in (BASE, BASE + '/tree/main/data'):
        body, record = get(url)
        assert b'comp1818-potential-shifting' in body
        checks.append(record)
    for relative in ('gridshift/environment.py', 'gridshift/learner.py',
                     'gridshift/potentials.py', 'gridshift/engine.py',
                     'data/environments.json', 'data/open25_goal.npz',
                     'data/open10_goal.npz', 'data/detour10_goal.npz'):
        body, record = get(RAW + relative)
        assert body == (ROOT/relative).read_bytes(), f'Published bytes differ: {relative}'
        record['matches_local_file'] = relative
        checks.append(record)
    if archive_url:
        assert archive_path, '--archive-path is needed for an exact local archive comparison'
        body, record = get(archive_url)
        assert body == Path(archive_path).read_bytes(), 'Downloaded archive differs from local archive'
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            assert archive.testzip() is None
            member = next(n for n in archive.namelist() if n.endswith('/SHA256SUMS.json'))
            prefix = member[:-len('SHA256SUMS.json')]
            manifest = json.loads(archive.read(member))
            for name, digest in manifest.items():
                assert hashlib.sha256(archive.read(prefix+name)).hexdigest() == digest, name
            record['verified_manifest_files'] = len(manifest)
        checks.append(record)
    result = {'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'passed', 'authentication': 'No GitHub token, Authorization header, cookie jar or owner session',
              'implementation_url': BASE, 'dataset_url': BASE+'/tree/main/data',
              'checks': checks}
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-url')
    parser.add_argument('--archive-path')
    parser.add_argument('--output', default='verification/public_access_r9.json')
    args = parser.parse_args()
    main(args.archive_url, args.archive_path, args.output)

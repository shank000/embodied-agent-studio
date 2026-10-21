"""Fetch the selected upstream URDFs and their referenced meshes, pinned by commit."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'EmbodiedStudio-model-setup'}), timeout=45) as response:
                return response.read()
        except (URLError, TimeoutError):
            if attempt == 2: raise
            time.sleep(attempt + 1)


def download(repo, revision, destination, urdf_path, extras):
    tree = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1'))
    paths = {item['path'] for item in tree['tree'] if item['type'] == 'blob'}
    base = f'https://raw.githubusercontent.com/{repo}/{revision}/'
    data = fetch(base + quote(urdf_path))
    urdf = ET.fromstring(data)
    selected = {urdf_path}
    for mesh in urdf.findall('.//mesh'):
        filename = mesh.get('filename')
        relative = filename.removeprefix('package://')
        matches = [relative] if relative in paths else [path for path in paths if path.endswith('/' + relative)]
        if not matches: matches = [path for path in paths if path.endswith('/' + filename.rsplit('/', 1)[-1])]
        if len(matches) != 1: raise ValueError(f'Ambiguous or missing mesh: {filename}: {matches}')
        selected.add(matches[0])
    selected.update(path for path in extras if path in paths)
    folder = ROOT / 'vendor' / destination
    def save(path):
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): target.write_bytes(data if path == urdf_path else fetch(base + quote(path)))
    with ThreadPoolExecutor(max_workers=5) as pool: list(pool.map(save, sorted(selected)))
    manifest = {'repository': f'https://github.com/{repo}', 'revision': revision,
                'urdf': urdf_path, 'files': sorted(selected)}
    (folder / 'SOURCE.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    joints = [(j.get('name'), j.get('type')) for j in urdf.findall('joint') if j.get('type') != 'fixed']
    print(json.dumps({'model': destination, 'revision': revision, 'joints': joints,
                      'files': len(selected)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    download('Stella-robot/GaiaHand', 'eb61f952ce0086a3fea4ecb6559970c5ea1f09a5', 'gaia_hand', 'Software/urdf/right_gaiahand/urdf/right_gaiahand.urdf',
             ['LICENSE', 'README_CN.md', 'Software/urdf/right_gaiahand/package.xml'])
    download('RealManRobot/ros2_rm_robot', 'c941b565e4f9174afa36561f143ef5fbbb744750', 'realman', 'rm_description/urdf/rm_65.urdf',
             ['LICENSE', 'LICENSE.txt', 'README.md', 'rm_description/package.xml'])

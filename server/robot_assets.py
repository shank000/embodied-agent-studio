"""Compose manufacturer URDFs using MuJoCo's native URDF importer."""
from functools import lru_cache
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import mujoco

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def rm65_gaia_source():
    assets = {}
    models = []
    for folder in ('realman', 'gaia_hand'):
        base = ROOT / 'vendor' / folder
        manifest = json.loads((base / 'SOURCE.json').read_text(encoding='utf-8'))
        root = ET.parse(base / manifest['urdf']).getroot()
        for mesh in root.findall('.//mesh'):
            original = mesh.get('filename').removeprefix('package://')
            candidates = [p for p in manifest['files'] if p == original or p.endswith('/' + original)]
            if not candidates: candidates = [p for p in manifest['files'] if p.endswith('/' + original.rsplit('/', 1)[-1])]
            if len(candidates) != 1: raise ValueError(f'无法唯一定位官方网格：{original}')
            name = folder + '_' + Path(original).name
            assets[name] = (base / candidates[0]).read_bytes()
            mesh.set('filename', name)
        models.append(root)
    arm, hand = models
    for element in hand: arm.append(element)
    mount = ET.SubElement(arm, 'joint', name='gaia_mount', type='fixed')
    ET.SubElement(mount, 'parent', link='Link6')
    ET.SubElement(mount, 'child', link='right_base_link')
    ET.SubElement(mount, 'origin', xyz='0 0 .012', rpy='0 0 0')
    extension = ET.SubElement(arm, 'mujoco')
    ET.SubElement(extension, 'compiler', discardvisual='false', fusestatic='false')
    model = mujoco.MjModel.from_xml_string(ET.tostring(arm, encoding='unicode'), assets)
    with tempfile.TemporaryDirectory(prefix='rm65-gaia-') as folder:
        path = str(Path(folder) / 'combined.xml')
        mujoco.mj_saveLastXML(path, model)
        root = ET.parse(path).getroot()
    root.set('model', 'RM65 + GaiaHand public 15-DOF URDF')
    for geom in root.findall('.//geom'):
        collision = geom.get('contype', '1') != '0'
        geom.set('group', '3' if collision else '2')
        if collision: geom.set('rgba', '.30 .50 .43 .45')
    actuator = ET.SubElement(root, 'actuator')
    hand_names = []
    for item in arm.findall('joint'):
        if item.get('type') == 'fixed': continue
        name = item.get('name'); finger = name.startswith('right_')
        limits = item.find('limit'); lower, upper = limits.get('lower'), limits.get('upper')
        effort = float(limits.get('effort', 0))
        # The public hand URDF leaves actuator effort/velocity at zero.
        # These are explicit simulation servo settings, not manufacturer calibration.
        if finger: effort = 1.0; hand_names.append(name)
        ET.SubElement(actuator, 'position', name=name + '_servo', joint=name,
                      kp='6' if finger else '1000', kv='.12' if finger else '60',
                      ctrlrange=f'{lower} {upper}', forcerange=f'{-effort} {effort}')
        joint = root.find(f".//joint[@name='{name}']")
        joint.set('damping', '.015' if finger else '.5')
        joint.set('armature', '.0001' if finger else '.02')
    palm = root.find(".//body[@name='right_base_link']")
    ET.SubElement(palm, 'site', name='studio_tcp', pos='0 0 .16', size='.007', rgba='.2 .7 .4 1', group='4')
    ET.SubElement(palm, 'camera', name='wrist_cam', pos='.045 0 .025', xyaxes='1 0 0 0 -1 0', fovy='75')
    # Adjacent URDF collision hulls overlap at joint sockets; parent contacts are
    # already filtered by MuJoCo, with all other robot/object contacts retained.
    return ET.tostring(root, encoding='unicode'), assets, hand_names


def rm65_gaia_xml():
    xml, assets, names = rm65_gaia_source()
    return ET.fromstring(xml), assets, names

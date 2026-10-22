"""Two and three real Panda instances sharing upstream assets and one physics world."""
from copy import deepcopy
import math
import xml.etree.ElementTree as ET


PANDA_BASES = (
    ((.50, -.68, 0.), math.pi / 2),
    ((.50, .68, 0.), -math.pi / 2),
    ((-.40, 0., 0.), 0.),
)


def assemble_pandas(root, count):
    """Clone bodies and their control constraints; meshes/defaults stay shared.

    The checked-in Panda uses these named references in body, tendon, equality,
    actuator and contact elements. Asset and default-class references intentionally
    keep their upstream names, so repeated arms do not duplicate mesh memory.
    """
    world = root.find('worldbody')
    body = world.find("body[@name='link0']")
    world.remove(body)
    sections = {}
    for tag in ('tendon', 'equality', 'actuator', 'contact'):
        section = root.find(tag)
        sections[tag] = list(section) if section is not None else []
        if section is None:
            section = ET.SubElement(root, tag)
        section[:] = []
    references = {'name', 'joint', 'joint1', 'joint2', 'tendon', 'body', 'body1', 'body2', 'site'}

    def clone(element, prefix):
        copy = deepcopy(element)
        for item in copy.iter():
            for attribute in references.intersection(item.attrib):
                item.set(attribute, prefix + item.get(attribute))
        return copy

    for index, (position, yaw) in enumerate(PANDA_BASES[:count], 1):
        prefix = f'arm{index}_'
        instance = clone(body, prefix)
        instance.set('pos', ' '.join(map(str, position)))
        instance.set('quat', f'{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}')
        world.append(instance)
        for tag, elements in sections.items():
            section = root.find(tag)
            section.extend(clone(element, prefix) for element in elements)

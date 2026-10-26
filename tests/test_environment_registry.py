"""Lightweight contracts for the selected simulator and task-suite integrations."""
import copy
import hashlib
import json
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from server import artifacts, research, simulators


class _WindowsWorkspace(PureWindowsPath):
    def resolve(self):
        return self


class EnvironmentRegistryTests(unittest.TestCase):
    def test_only_requested_and_previously_integrated_engines_are_registered(self):
        expected = {'mujoco', 'genesis', 'maniskill', 'libero', 'alfworld', 'alfworld_thor', 'ai2thor', 'isaac', 'isaacsim', 'robocasa365'}
        self.assertEqual(set(simulators.EVALUATORS), expected)
        self.assertEqual(set(simulators.RUNTIMES), expected-{'mujoco'})
        for unsupported in ('omnigibson', 'habitat', 'pybullet'):
            with self.subTest(engine=unsupported), self.assertRaises(ValueError):
                simulators.runtime_python(unsupported)

    def test_catalog_has_real_task_metadata_without_starting_optional_sdks(self):
        sdk_modules = {'genesis', 'mani_skill', 'sapien', 'libero', 'alfworld', 'ai2thor', 'isaaclab', 'isaacsim', 'newton'}
        before = {name for name in sys.modules if name.split('.')[0] in sdk_modules}
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('Catalog must not launch SDKs')):
            catalog = simulators.catalog()
            self.assertEqual({row['id'] for row in catalog['backends']}, set(simulators.EVALUATORS)|{'robodojo'})
            for row in catalog['backends']:
                if row['id']=='robodojo':
                    self.assertEqual(row['kind'],'benchmark')
                    self.assertNotIn('load',row['capabilities'])
                    continue
                models, scenes, tasks = simulators.supported_catalog(row['id'])
                self.assertEqual(set(row['models']), set(models))
                self.assertEqual(set(row['scenes']), set(scenes))
                self.assertTrue(row['implemented'])
                for scene in scenes.values():
                    self.assertTrue(scene['tasks'])
                    self.assertTrue(all(task in tasks for task in scene['tasks']))
        after = {name for name in sys.modules if name.split('.')[0] in sdk_modules}
        self.assertEqual(after, before, 'Host metadata queries must not import native optional SDK packages')

    def test_all_catalog_tasks_validate_and_research_registration_matches_action_support(self):
        with tempfile.TemporaryDirectory(prefix='environment-protocols-') as folder:
            with patch.object(research, 'PROTOCOLS', Path(folder)):
                for engine in ('maniskill', 'libero', 'alfworld', 'alfworld_thor', 'ai2thor', 'isaac', 'isaacsim'):
                    models, scenes, tasks = simulators.supported_catalog(engine)
                    seen = set()
                    for scene_id, scene in scenes.items():
                        for task_id in scene['tasks']:
                            with self.subTest(engine=engine, scene=scene_id, task=task_id):
                                spec = {'name': f'{engine}/{scene_id}/{task_id}',
                                    'engine_id': engine, 'robot': next(iter(models)), 'scene_id': scene_id, 'task_id': task_id}
                                research.ResearchProtocol.model_validate(spec)
                                if simulators.backend_identity(engine)['action_mode'] == 'tcp_pose':
                                    protocol = research.register_protocol(spec)
                                    loaded = research.get_protocol(protocol['protocol_id'])
                                    self.assertEqual(loaded['task'], tasks[task_id])
                                    self.assertEqual(loaded['spec']['engine_id'], engine)
                                    contract = loaded['contract']
                                else:
                                    # Discrete tasks remain selectable and evaluable, but
                                    # cannot register a TCP-only research execution plan.
                                    before = set(Path(folder).iterdir())
                                    with self.assertRaisesRegex(ValueError, '普通 Agent'):
                                        research.register_protocol(spec)
                                    self.assertEqual(set(Path(folder).iterdir()), before)
                                    contract = research.evaluation_contract(engine)
                                self.assertEqual(contract['backend']['evaluator_source'], simulators.EVALUATORS[engine][0])
                                seen.add(task_id)
                    self.assertEqual(seen, set(tasks))

    def test_protocol_task_description_is_frozen_after_catalog_metadata_changes(self):
        models, scenes, tasks = simulators.supported_catalog('libero')
        with tempfile.TemporaryDirectory(prefix='frozen-environment-task-') as folder:
            with patch.object(research, 'PROTOCOLS', Path(folder)):
                protocol = research.register_protocol({'name': 'Official drawer goal', 'engine_id': 'libero',
                    'robot': 'panda', 'scene_id': 'libero_goal', 'task_id': 'libero_goal:0'})
                changed = copy.deepcopy(tasks)
                changed['libero_goal:0']['instruction'] = 'Changed current catalog text'
                with patch.object(research, 'supported_catalog', return_value=(models, scenes, changed)):
                    loaded = research.get_protocol(protocol['protocol_id'])
                self.assertEqual(loaded['task']['instruction'], 'open the middle drawer of the cabinet')
                self.assertEqual(loaded['task'], protocol['task'])

    def test_thor_protocol_freezes_shared_discrete_environment_source(self):
        from server.backends.alfworld_backend import DiscreteEnvironment
        from server.backends.thor_backend import ThorSimulation
        self.assertTrue(issubclass(ThorSimulation, DiscreteEnvironment))
        contract = research.evaluation_contract('ai2thor')
        name = 'server/backends/alfworld_backend.py'
        self.assertEqual(contract['files'][name], hashlib.sha256((research.ROOT/name).read_bytes()).hexdigest())
        self.assertEqual(contract['input_condition'], 'native_rgb_visible_metadata')
        self.assertEqual(contract['inference_timing'], 'turn_based')

    def test_wsl_marker_and_fixed_argv_preserve_paths_without_a_shell(self):
        marker_data = {'kind': 'wsl', 'distribution': 'Ubuntu', 'python': '/home/researcher/thor env/bin/python3.11',
                       'validated': True, 'assets_ready': True, 'version': '5.0.0'}
        with tempfile.TemporaryDirectory(prefix='wsl-runtime-contract-') as folder:
            root = Path(folder)
            marker = root/'.venv-ai2thor/runtime.json'
            marker.parent.mkdir()
            marker.write_text(json.dumps(marker_data), encoding='utf-8')
            with patch.object(simulators, 'ROOT', root), patch.object(simulators.sys, 'platform', 'win32'):
                self.assertTrue(simulators.runtime_info('ai2thor')['installed'])
        with patch.object(simulators, 'runtime_info', return_value=marker_data|{'installed': True}), \
                patch.object(simulators, 'ROOT', _WindowsWorkspace('D:/robot work/具身平台')):
            command = simulators.runtime_command('ai2thor', generation_base=17)
        self.assertEqual(command, ['wsl.exe', '--distribution', 'Ubuntu', '--cd', '/mnt/d/robot work/具身平台',
            '--exec', marker_data['python'], '-u', '-m', 'server.backends.runtime', '--engine', 'ai2thor', '--generation-base', '17'])
        self.assertFalse(any(value in command for value in ('bash', 'sh', '-c', 'cmd', '/c')))

    def test_alfworld_visual_contract_is_distinct_and_freezes_shared_native_code(self):
        text = research.evaluation_contract('alfworld')
        visual = research.evaluation_contract('alfworld_thor')
        self.assertEqual(text['input_condition'], 'native_text')
        self.assertEqual(visual['input_condition'], 'native_rgb_visible_metadata')
        self.assertEqual(visual['backend']['action_mode'], 'discrete')
        self.assertEqual(visual['backend']['state_schema'], 'alfworld-thor-lowlevel-v1')
        self.assertNotEqual(text['backend']['evaluator_source'], visual['backend']['evaluator_source'])
        for name in ('alfworld_backend.py', 'thor_backend.py', 'alfworld_thor_assets.py'):
            self.assertIn('server/backends/'+name, visual['files'])
        self.assertNotEqual(simulators.runtime_python('alfworld'), simulators.runtime_python('alfworld_thor'))

    def test_invalid_wsl_markers_are_rejected_before_launch(self):
        valid = {'kind': 'wsl', 'distribution': 'Ubuntu', 'python': '/home/researcher/env/bin/python3.11',
                 'validated': True, 'assets_ready': True}
        invalid = [[], None, valid|{'kind': 'shell'}, valid|{'distribution': '--exec'},
            valid|{'distribution': 'Ubuntu\nsh'}, valid|{'python': 'relative/python'},
            valid|{'python': '/bin/sh'}, valid|{'python': '/env/bin/python3.11\n--bad'},
            valid|{'validated': 'false'}, valid|{'assets_ready': 'false'}]
        with tempfile.TemporaryDirectory(prefix='invalid-wsl-marker-') as folder:
            root = Path(folder)
            marker = root/'.venv-ai2thor/runtime.json'
            marker.parent.mkdir()
            with patch.object(simulators, 'ROOT', root), patch.object(simulators.sys, 'platform', 'win32'):
                for value in invalid:
                    with self.subTest(value=value):
                        marker.write_text(json.dumps(value), encoding='utf-8')
                        with self.assertRaises(ValueError): simulators.runtime_info('ai2thor')
                marker.write_text(json.dumps(valid|{'assets_ready': False}), encoding='utf-8')
                self.assertFalse(simulators.runtime_info('ai2thor')['installed'])

    def test_method_materials_include_every_simulator_dependency_spec(self):
        footprint, blobs = artifacts.version_materials()
        expected = {path.name for path in artifacts.ROOT.glob('requirements*.txt')}
        self.assertTrue({'requirements-libero.txt', 'requirements-maniskill.txt', 'requirements-alfworld.txt',
                         'requirements-thor.txt', 'requirements-isaac.txt'}.issubset(expected))
        self.assertTrue(expected.issubset(blobs))
        for name in expected:
            self.assertEqual(footprint['files'][name]['sha256'], hashlib.sha256(blobs[name]).hexdigest())

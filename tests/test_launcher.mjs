// Only Python's standard library is invoked; no server or simulator is started.
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {copyFileSync, mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {venvPython} from '../tools/python.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const launcher = path.join(root, 'tools', 'python.mjs');

test('selects the project virtual environment for Windows, Linux, and macOS', () => {
  assert.equal(venvPython(root, 'win32'), path.join(root, '.venv', 'Scripts', 'python.exe'));
  for (const platform of ['linux', 'darwin']) {
    assert.equal(venvPython(root, platform), path.join(root, '.venv', 'bin', 'python'));
  }
});

test('uses the project environment and cwd, preserving literal arguments and exit status', () => {
  const temporary = mkdtempSync(path.join(os.tmpdir(), 'workbench-launcher-'));
  try {
    const argument = 'space 中文 ; $HOME & "quoted"';
    const script = 'import json,os,sys;print(json.dumps({"cwd":os.getcwd(),"prefix":sys.prefix,"args":sys.argv[1:]}));sys.exit(7)';
    const result = spawnSync(process.execPath, [launcher, '-c', script, argument], {cwd:temporary, encoding:'utf8'});
    assert.equal(result.status, 7, result.stderr);
    const output = JSON.parse(result.stdout);
    assert.equal(realpathSync(output.cwd), realpathSync(root));
    assert.equal(realpathSync(output.prefix), realpathSync(path.join(root, '.venv')));
    assert.deepEqual(output.args, [argument]);
  } finally {
    rmSync(temporary, {recursive:true, force:true});
  }
});

test('missing environment fails with setup instructions instead of using system Python', () => {
  const temporary = mkdtempSync(path.join(os.tmpdir(), 'workbench-launcher-missing-'));
  try {
    mkdirSync(path.join(temporary, 'tools'));
    const copy = path.join(temporary, 'tools', 'python.mjs');
    copyFileSync(launcher, copy);
    const result = spawnSync(process.execPath, [copy, '-c', 'print("unexpected")'], {encoding:'utf8'});
    assert.equal(result.status, 1);
    assert.equal(result.stdout, '');
    assert.match(result.stderr, /Project Python environment not found:/);
    assert.ok(result.stderr.includes(venvPython(realpathSync(temporary))));
    assert.match(result.stderr, /-m venv \.venv/);
    assert.match(result.stderr, /-m pip install -r requirements\.txt/);
  } finally {
    rmSync(temporary, {recursive:true, force:true});
  }
});

test('direct execution through a linked directory still reports a missing environment', () => {
  const temporary = mkdtempSync(path.join(os.tmpdir(), 'workbench-launcher-linked-'));
  try {
    const project = path.join(temporary, 'project');
    const linked = path.join(temporary, 'linked');
    mkdirSync(path.join(project, 'tools'), {recursive:true});
    copyFileSync(launcher, path.join(project, 'tools', 'python.mjs'));
    symlinkSync(project, linked, process.platform === 'win32' ? 'junction' : 'dir');
    const result = spawnSync(process.execPath, [path.join(linked, 'tools', 'python.mjs'), '-c', 'print("unexpected")'], {encoding:'utf8'});
    assert.equal(result.status, 1, result.stderr);
    assert.equal(result.stdout, '');
    assert.ok(result.stderr.includes(venvPython(realpathSync(project))));
  } finally {
    rmSync(temporary, {recursive:true, force:true});
  }
});

test('importing the launcher from stdin does not execute it', () => {
  const script = `await import(${JSON.stringify(pathToFileURL(launcher).href)}); console.log('imported');`;
  const result = spawnSync(process.execPath, ['--input-type=module', '-'], {input:script, encoding:'utf8'});
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout, 'imported\n');
  assert.equal(result.stderr, '');
});

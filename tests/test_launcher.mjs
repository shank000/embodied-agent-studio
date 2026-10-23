// Only Python's standard library is invoked; no server or simulator is started.
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {copyFileSync, mkdirSync, mkdtempSync, realpathSync, rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
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
    assert.ok(result.stderr.includes(venvPython(temporary)));
    assert.match(result.stderr, /-m venv \.venv/);
    assert.match(result.stderr, /-m pip install -r requirements\.txt/);
  } finally {
    rmSync(temporary, {recursive:true, force:true});
  }
});

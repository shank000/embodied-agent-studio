// Run the project's Python environment without relying on shell activation.
import {spawnSync} from 'node:child_process';
import {existsSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

export function venvPython(root, platform=process.platform) {
  return path.join(root, '.venv', ...(platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python']));
}

export function main(args) {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const python = venvPython(root);
  if (!existsSync(python)) {
    const create = process.platform === 'win32' ? 'py -3 -m venv .venv' : 'python3 -m venv .venv';
    const install = process.platform === 'win32' ? '.venv\\Scripts\\python.exe' : '.venv/bin/python';
    console.error(`Project Python environment not found: ${python}\nRun from the project directory:\n  ${create}\n  ${install} -m pip install -r requirements.txt`);
    return 1;
  }
  if (!args.length) {
    console.error('Usage: node tools/python.mjs <Python arguments>');
    return 2;
  }
  const result = spawnSync(python, args, {cwd: root, stdio: 'inherit', windowsHide: true});
  if (result.error) {
    console.error(`Unable to run project Python (${python}): ${result.error.message}`);
    return 1;
  }
  return result.status ?? (result.signal === 'SIGINT' ? 130 : result.signal === 'SIGTERM' ? 143 : 1);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main(process.argv.slice(2));
}

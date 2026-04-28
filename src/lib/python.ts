import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { execa } from 'execa';
import ffmpegStatic from 'ffmpeg-static';
import { log } from './logger.js';
import { PYTHON_DIR, VENV_PYTHON } from './paths.js';

export type PythonProgress = (data: string) => void;

export async function runPython(
  scriptRelPath: string, 
  args: string[], 
  opts?: { cwd?: string, onProgress?: PythonProgress }
): Promise<void> {
  if (!existsSync(VENV_PYTHON)) {
    throw new Error('Run `pnpm setup` first.');
  }

  const scriptAbsPath = join(PYTHON_DIR, scriptRelPath);

  // Find ffmpeg path to inject into PATH
  const env = { ...process.env };
  if (ffmpegStatic) {
    const ffmpegDir = dirname(ffmpegStatic);
    env.PATH = `${ffmpegDir}${process.platform === 'win32' ? ';' : ':'}${env.PATH}`;
    log.debug(`Injected FFmpeg directory into PATH: ${ffmpegDir}`);
  } else {
    log.warn('ffmpeg-static binary not found. Python scripts might fail to find ffmpeg.');
  }

  const subprocess = execa(VENV_PYTHON, [scriptAbsPath, ...args], {
    cwd: opts?.cwd ?? PYTHON_DIR,
    stdio: opts?.onProgress ? ['inherit', 'pipe', 'inherit'] : 'inherit',
    env: {
      ...env,
      PYTHONUNBUFFERED: '1',
      PYTHONIOENCODING: 'utf-8',
    },
  });

  if (opts?.onProgress && subprocess.stdout) {
    subprocess.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      opts.onProgress!(text);
    });
  }

  try {
    await subprocess;
  } catch (error: any) {
    throw new Error(`Python script failed with exit code ${error.exitCode}`);
  }
}

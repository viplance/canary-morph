import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { existsSync, createWriteStream } from 'node:fs';
import { join } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { execa } from 'execa';
import { log } from '../lib/logger.js';
import {
  DATASET_DIR,
  MODELS_DIR,
  PRETRAINED_DIR,
  PYTHON_DIR,
  SOURCE_DIR,
  TMP_DIR,
  TRAINED_DIR,
  VENV_DIR,
  VENV_PYTHON,
} from '../lib/paths.js';

const WEIGHTS = [
  {
    name: 'hubert_base.pt',
    url: 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt',
  },
  {
    name: 'rmvpe.pt',
    url: 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt',
  },
  {
    name: 'f0G48k.pth',
    url: 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth',
  },
  {
    name: 'f0D48k.pth',
    url: 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth',
  },
];

async function downloadFile(url: string, dest: string) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to download ${url}: ${response.statusText}`);
  if (!response.body) throw new Error(`No body in response for ${url}`);
  
  const writer = createWriteStream(dest);
  await pipeline(response.body as any, writer);
}

export async function runSetup() {
  log.info('Starting setup...');

  // 1. Ensure directories
  const dirs = [SOURCE_DIR, MODELS_DIR, PRETRAINED_DIR, DATASET_DIR, TRAINED_DIR, TMP_DIR, PYTHON_DIR];
  for (const dir of dirs) {
    if (!existsSync(dir)) {
      await mkdir(dir, { recursive: true });
    }
  }

  // 2. Verify Python 3.10
  try {
    const { stdout } = await execa('python3.10', ['--version']);
    log.info(`Using ${stdout}`);
  } catch {
    throw new Error('Install Python 3.10.x (e.g., `brew install python@3.10` or `pyenv install 3.10.14`)');
  }

  // 3. Create venv
  if (!existsSync(VENV_DIR)) {
    log.info('Creating virtual environment...');
    await execa('python3.10', ['-m', 'venv', VENV_DIR]);
  }

  // 4. Upgrade pip (Downgrade to <24.1 for fairseq compatibility)
  log.info('Updating pip (downgrading to <24.1 for compatibility)...');
  await execa(VENV_PYTHON, ['-m', 'pip', 'install', 'pip<24.1', 'wheel', 'setuptools']);

  // 5. Install requirements
  log.info('Installing Python requirements...');
  // Force omegaconf first
  await execa(VENV_PYTHON, ['-m', 'pip', 'install', 'omegaconf==2.0.6']);
  
  let reqPath = join(PYTHON_DIR, 'requirements.txt');
  if (process.platform === 'darwin') {
    const content = await readFile(reqPath, 'utf8');
    const filtered = content.split('\n').filter(line => !line.includes('--extra-index-url')).join('\n');
    reqPath = join(TMP_DIR, 'requirements.darwin.txt');
    await writeFile(reqPath, filtered);
  }
  await execa(VENV_PYTHON, ['-m', 'pip', 'install', '-r', reqPath]);

  // 6. Download pretrained weights
  for (const weight of WEIGHTS) {
    const dest = join(PRETRAINED_DIR, weight.name);
    if (existsSync(dest)) {
      log.info(`Skipping ${weight.name} (already exists)`);
      continue;
    }
    log.info(`Downloading ${weight.name}...`);
    await downloadFile(weight.url, dest);
  }

  // 7. Clone RVC-Project for training scripts
  const rvcSrcDir = join(MODELS_DIR, 'rvc-src');
  if (!existsSync(rvcSrcDir)) {
    log.info('Cloning RVC-Project...');
    await execa('git', ['clone', 'https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI', rvcSrcDir]);
  }

  log.success('Setup complete.');
}

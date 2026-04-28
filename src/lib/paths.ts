import { join } from 'node:path';

export const ROOT = process.cwd();
export const SOURCE_DIR = join(ROOT, 'source');
export const MODELS_DIR = join(ROOT, 'models');
export const PRETRAINED_DIR = join(ROOT, 'models/pretrained');
export const DATASET_DIR = join(ROOT, 'models/dataset');
export const TRAINED_DIR = join(ROOT, 'models/trained');
export const TMP_DIR = join(ROOT, 'tmp');
export const PYTHON_DIR = join(ROOT, 'python');
export const VENV_DIR = join(ROOT, 'python/.venv');

export const VENV_PYTHON = process.platform === 'win32' 
  ? join(VENV_DIR, 'Scripts/python.exe') 
  : join(VENV_DIR, 'bin/python');

export const MODEL_NAME = 'canary';
export const MODEL_PTH = join(TRAINED_DIR, `${MODEL_NAME}.pth`);
export const MODEL_INDEX = join(TRAINED_DIR, `${MODEL_NAME}.index`);

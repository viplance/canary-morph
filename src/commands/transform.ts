import { existsSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { join } from 'node:path';
import { log } from '../lib/logger.js';
import { runPython } from '../lib/python.js';
import { decodeForRVC, encodeFinal } from '../lib/ffmpeg.js';
import {
  MODEL_INDEX,
  MODEL_PTH,
  TMP_DIR,
} from '../lib/paths.js';

export interface TransformOptions {
  pitch?: number;
  method?: 'rmvpe' | 'pm' | 'harvest' | 'crepe';
  indexRate?: number;
  protect?: number;
  rmsMixRate?: number;
  filterRadius?: number;
  device?: 'auto' | 'cpu' | 'mps' | 'cuda';
  bitrate?: number;
}

export async function runTransform(input: string, output: string, opts: TransformOptions) {
  // 1. Validate input
  if (!existsSync(input)) {
    throw new Error(`Input file not found: ${input}`);
  }
  if (!output.endsWith('.wav') && !output.endsWith('.mp3')) {
    throw new Error('Output file must end with .wav or .mp3');
  }

  // 2. Validate model
  if (!existsSync(MODEL_PTH) || !existsSync(MODEL_INDEX)) {
    throw new Error('Train the model first: pnpm train');
  }

  const id = Math.random().toString(36).substring(7);
  const in16k = join(TMP_DIR, `${id}.in16k.wav`);
  const out48kFloat = join(TMP_DIR, `${id}.out48k.wav`);

  try {
    log.info('Decoding input audio...');
    await decodeForRVC(input, in16k);

    log.info('Running voice conversion...');
    await runPython('conversion.py', [
      '--input', in16k,
      '--output', out48kFloat,
      '--model', MODEL_PTH,
      '--index', MODEL_INDEX,
      '--pitch', String(opts.pitch ?? 0),
      '--method', opts.method ?? 'rmvpe',
      '--index-rate', String(opts.indexRate ?? 0.75),
      '--protect', String(opts.protect ?? 0.33),
      '--rms-mix-rate', String(opts.rmsMixRate ?? 0.25),
      '--filter-radius', String(opts.filterRadius ?? 3),
      '--device', opts.device ?? 'auto',
    ]);

    const fmt = output.endsWith('.mp3') ? `MP3 ${opts.bitrate ?? 192}kbps` : '48kHz 24-bit WAV';
    log.info(`Encoding final output (${fmt})...`);
    await encodeFinal(out48kFloat, output, opts.bitrate);

    log.success(`Transformation complete: ${output}`);
  } finally {
    if (existsSync(in16k)) await rm(in16k);
    if (existsSync(out48kFloat)) await rm(out48kFloat);
  }
}

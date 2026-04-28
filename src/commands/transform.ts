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

export async function runTransform(input: string, output: string, opts: { pitch?: number }) {
  // 1. Validate input
  if (!existsSync(input)) {
    throw new Error(`Input file not found: ${input}`);
  }
  if (!output.endsWith('.wav')) {
    throw new Error('Output file must end with .wav');
  }

  // 2. Validate model
  if (!existsSync(MODEL_PTH) || !existsSync(MODEL_INDEX)) {
    throw new Error('Train the model first: pnpm train');
  }

  const id = Math.random().toString(36).substring(7);
  const in16k = join(TMP_DIR, `${id}.in16k.wav`);
  const out48kFloat = join(TMP_DIR, `${id}.out48k.wav`);

  try {
    // 5. Decode for RVC
    log.info('Decoding input audio...');
    await decodeForRVC(input, in16k);

    // 6. Run Python inference
    log.info('Running voice conversion...');
    await runPython('conversion.py', [
      '--input', in16k,
      '--output', out48kFloat,
      '--model', MODEL_PTH,
      '--index', MODEL_INDEX,
      '--pitch', String(opts.pitch ?? 0),
      '--method', 'rmvpe',
    ]);

    // 7. Encode final
    log.info('Encoding final output (48kHz, 24-bit, mono)...');
    await encodeFinal(out48kFloat, output);

    log.success(`Transformation complete: ${output}`);
  } finally {
    // 8. Cleanup
    if (existsSync(in16k)) await rm(in16k);
    if (existsSync(out48kFloat)) await rm(out48kFloat);
  }
}

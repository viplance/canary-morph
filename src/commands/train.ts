import { existsSync, mkdirSync, symlinkSync } from 'node:fs';
import { readdir } from 'node:fs/promises';
import { join } from 'node:path';
import * as cliProgress from 'cli-progress';
import { log } from '../lib/logger.js';
import { runPython } from '../lib/python.js';
import { runSetup } from './setup.js';
import {
  DATASET_DIR,
  MODEL_INDEX,
  MODEL_NAME,
  MODEL_PTH,
  MODELS_DIR,
  PRETRAINED_DIR,
  SOURCE_DIR,
  TRAINED_DIR,
  VENV_PYTHON,
} from '../lib/paths.js';

export async function runTrain(opts: { epochs?: number; batchSize?: number; reprep?: boolean }) {
  if (!existsSync(VENV_PYTHON)) {
    log.info('Python environment not found. Running setup...');
    await runSetup();
  }

  // 1. Ensure RVC assets are symlinked correctly
  const rvcSrcDir = join(MODELS_DIR, 'rvc-src');
  const rvcAssetsDir = join(rvcSrcDir, 'assets');
  
  const assetLinks = [
    { src: join(PRETRAINED_DIR, 'hubert_base.pt'), dest: join(rvcAssetsDir, 'hubert', 'hubert_base.pt') },
    { src: join(PRETRAINED_DIR, 'rmvpe.pt'), dest: join(rvcAssetsDir, 'rmvpe', 'rmvpe.pt') },
  ];

  for (const link of assetLinks) {
    const destDir = join(link.dest, '..');
    if (!existsSync(destDir)) mkdirSync(destDir, { parents: true });
    if (!existsSync(link.dest)) {
      try {
        symlinkSync(link.src, link.dest);
      } catch (e) {
        // Fallback to copy if symlink fails
        import('node:fs').then(fs => fs.copyFileSync(link.src, link.dest));
      }
    }
  }

  const sourceFiles = await readdir(SOURCE_DIR);
  const audioFiles = sourceFiles.filter(f => /\.(wav|mp3|flac|m4a|ogg)$/i.test(f));
  if (audioFiles.length === 0) {
    throw new Error('Place voice samples in ./source/ (.wav/.mp3/.flac/.m4a/.ogg)');
  }

  log.info(`Found ${audioFiles.length} audio files in source. Starting training pipeline...`);

  const multibar = new cliProgress.MultiBar({
    clearOnComplete: false,
    hideCursor: true,
    format: '{bar} | {percentage}% | {value}/{total} | {stage} | ETA: {eta}s',
  }, cliProgress.Presets.shades_classic);

  let currentBar: cliProgress.SingleBar | null = null;

  const onProgress = (data: string) => {
    // Stage updates
    if (data.includes('start preprocess')) {
        currentBar = multibar.create(audioFiles.length, 0, { stage: 'Slicing' });
    }
    if (data.includes('Success')) {
        currentBar?.increment();
    }

    const match = data.match(/now-(\d+),all-(\d+)/);
    if (match) {
      const current = parseInt(match[1], 10);
      const total = parseInt(match[2], 10);
      
      if (!currentBar || currentBar.getTotal() !== total) {
        let stage = 'Extracting';
        if (data.includes('f0')) stage = 'F0 Extract';
        else if (data.includes('feature')) stage = 'HuBERT';
        
        currentBar = multibar.create(total, 0, { stage });
      }
      currentBar.update(current);
    }
  };

  try {
    // Stage 1-4: Preprocess, F0, Feature, Train
    await runPython('train.py', [
      '--source', SOURCE_DIR,
      '--dataset', DATASET_DIR,
      '--pretrained', PRETRAINED_DIR,
      '--out', TRAINED_DIR,
      '--name', MODEL_NAME,
      '--epochs', String(opts.epochs ?? 200),
      '--batch-size', String(opts.batchSize ?? 4),
      '--sample-rate', '48000',
      ...(opts.reprep ? ['--reprep'] : []),
    ], { 
      onProgress
    });

    multibar.stop();
    log.success('Pipeline finished.');

    if (!existsSync(MODEL_PTH) || !existsSync(MODEL_INDEX)) {
      log.warn('Trained model files not found in expected location. Check models/trained/');
    } else {
      log.success(`Training complete! Model saved to ${MODEL_PTH}`);
    }
  } catch (err) {
    multibar.stop();
    throw err;
  }
}

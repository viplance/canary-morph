import { Command, InvalidArgumentError } from 'commander';
import { runSetup } from './commands/setup.js';
import { runTrain } from './commands/train.js';
import { runTransform } from './commands/transform.js';

const parseInt10 = (v: string) => {
  const n = parseInt(v, 10);
  if (Number.isNaN(n)) throw new InvalidArgumentError('Not a valid integer.');
  return n;
};

const parseFloatBounded = (min: number, max: number) => (v: string) => {
  const n = parseFloat(v);
  if (Number.isNaN(n)) throw new InvalidArgumentError('Not a valid number.');
  if (n < min || n > max) {
    throw new InvalidArgumentError(`Must be between ${min} and ${max}.`);
  }
  return n;
};

const program = new Command();

program
  .name('canary')
  .description('Voice timbre transformation (RVC v2)')
  .version('0.1.0');

program
  .command('setup')
  .description('Install Python deps and download pretrained models')
  .action(runSetup);

program
  .command('train')
  .description('Train a voice model from samples in ./source/')
  .option('-e, --epochs <n>', 'training epochs', parseInt10, 200)
  .option('-b, --batch-size <n>', 'batch size (raise to 8 if >12 GB VRAM)', parseInt10, 4)
  .option('--save-every <n>', 'checkpoint frequency in epochs', parseInt10, 50)
  .option('--top-db <db>', 'silence threshold for slicing (higher = more aggressive trim, denser data)', parseInt10, 30)
  .option('--device <name>', 'auto | cpu | mps | cuda', 'auto')
  .option('--cache-in-gpu', 'keep dataset in GPU memory (CUDA only, >12 GB VRAM)', false)
  .option('--reprep', 're-run dataset preparation (use after editing ./source/)', false)
  .action((opts) => runTrain(opts));

program
  .command('transform <input> <output>')
  .description('Convert <input> audio file to the trained timbre, write to <output>.wav')
  .option('-p, --pitch <semitones>', 'pitch shift in semitones (e.g. 12 male->female, -12 female->male)', parseInt10, 0)
  .option('--method <name>', 'pitch extractor: rmvpe | pm | harvest | crepe', 'rmvpe')
  .option('--index-rate <0-1>', 'how strongly to retrieve reference timbre (0=off, 1=max)', parseFloatBounded(0, 1), 0.75)
  .option('--protect <0-0.5>', 'protect unvoiced consonants (lower = more conversion, more artifacts)', parseFloatBounded(0, 0.5), 0.33)
  .option('--rms-mix-rate <0-1>', 'blend reference loudness envelope (0=keep source dynamics, 1=match reference)', parseFloatBounded(0, 1), 0.25)
  .option('--filter-radius <n>', 'pitch median-filter radius (odd 0-7; smooths f0 jitter)', parseInt10, 3)
  .option('--device <name>', 'auto | cpu | mps | cuda', 'auto')
  .action((input, output, opts) => runTransform(input, output, opts));

program.parseAsync(process.argv).catch((err) => {
  console.error(err?.message ?? err);
  process.exit(1);
});

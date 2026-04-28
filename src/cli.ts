import { Command } from 'commander';
import { runSetup } from './commands/setup.js';
import { runTrain } from './commands/train.js';
import { runTransform } from './commands/transform.js';

const program = new Command();

program
  .name('canary')
  .description('Voice timbre transformation')
  .version('0.1.0');

program
  .command('setup')
  .description('Install Python deps and download pretrained models')
  .action(runSetup);

program
  .command('train')
  .option('-e, --epochs <n>', 'training epochs', (v) => parseInt(v, 10), 200)
  .option('-b, --batch-size <n>', 'batch size', (v) => parseInt(v, 10), 4)
  .option('--reprep', 're-run dataset preparation', false)
  .action((opts) => runTrain(opts));

program
  .command('transform <input> <output>')
  .option('-p, --pitch <semitones>', 'pitch shift in semitones', (v) => parseInt(v, 10), 0)
  .action((input, output, opts) => runTransform(input, output, opts));

program.parseAsync(process.argv).catch((err) => {
  console.error(err?.message ?? err);
  process.exit(1);
});

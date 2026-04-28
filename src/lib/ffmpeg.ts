import ffmpegStatic from 'ffmpeg-static';
import ffmpeg from 'fluent-ffmpeg';

if (ffmpegStatic) {
  ffmpeg.setFfmpegPath(ffmpegStatic);
}

export async function decodeForRVC(inputPath: string, outWavPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .audioChannels(1)
      .audioFrequency(16000)
      .audioCodec('pcm_s16le')
      .format('wav')
      .on('end', () => resolve())
      .on('error', (err) => reject(err))
      .save(outWavPath);
  });
}

export async function encodeFinal(inputPath: string, outputPath: string): Promise<void> {
  // ffmpeg-static is built without libsoxr, so the soxr resampler option from the
  // spec is unavailable. The input is already 48k mono float; the only work here
  // is bit-depth conversion (float32 -> 24-bit PCM), no resampling required.
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .audioChannels(1)
      .audioFrequency(48000)
      .audioCodec('pcm_s24le')
      .format('wav')
      .on('end', () => resolve())
      .on('error', (err) => reject(err))
      .save(outputPath);
  });
}

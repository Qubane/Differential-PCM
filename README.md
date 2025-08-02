# Differential-PCM
Simplistic DPCM compression algorithm, with parabolic difference mapping

# Supported file types
- only .wav files

## Supported wave files
- supports 8 and 16 bit sample widths (although, don't expect higher quality from 16 bit, as it's treated as 8 bit)
- supports multi-track audio
- supports only signed audio samples

# Usage
- `py3.12 main.py -h` to get help
- `-i` / `--input` - input `.wav` file or `.dpcm` compressed file
- `-o` / `--output` - output `.wav` file or `.dpcm` compressed file
- `--mode` - mode of operation
  - `encode_wav` - encodes `.wav` to `.dpcm`
  - `decode_wav` - decodes `.dpcm` to `.wav`
  - `squezee` - encodes and then decodes `.wav` file. Used for quality preview
- `--dpcm-depth` - compression bit depth, 1 makes worst quality at 1/8th file size, 4 makes "best" quality at 1/2 file size

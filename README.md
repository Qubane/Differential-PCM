# Differential-PCM
Simplistic DPCM compression algorithm, with parabolic difference mapping

# Supported file types
- only .wav files

# Usage
- `py3.12 main.py -h` to get help
- `-i` / `--input` - input `.wav` file or `.dpcm` compressed file
- `-o` / `--output` - output `.wav` file or `.dpcm` compressed file
- `--mode` - mode of operation
  - `encode_wav` - encodes `.wav` to `.dpcm`
  - `decode_wav` - decodes `.dpcm` to `.wav`
  - `squezee` - encodes and then decodes `.wav` file. Used for quality preview

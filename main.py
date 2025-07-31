import wave
import struct
import argparse


DPCM_SIZE = 16


def _dpcm_mapping(x):
    value = int(abs(x) * 1) + 1
    if x < 0:
        return -value
    return value


DPCM_MAP = [_dpcm_mapping(x + 0.5) for x in range(-DPCM_SIZE // 2, DPCM_SIZE // 2)]
print(DPCM_MAP)


def dpcm_quantize(value: int) -> int:
    """
    Quantize the value
    """

    if value > 0:
        for idx, mapping in enumerate(DPCM_MAP[8:]):
            if value - mapping <= 0:
                return idx + 8
        return len(DPCM_MAP) - 1
    else:
        for idx, mapping in enumerate(DPCM_MAP[8:]):
            if value + mapping <= 0:
                return idx
        return 0


def dpcm_encode(samples: list[int]) -> list[int]:
    """
    Encodes samples using DPCM
    """

    encoded_samples = []
    previous_sample = samples[0]

    for sample in samples[1:]:
        # calculate difference
        diff = sample - previous_sample

        # map to range
        quantized_diff = dpcm_quantize(diff)

        # append to encoded samples
        encoded_samples.append(quantized_diff)

        previous_sample += DPCM_MAP[quantized_diff]

    return encoded_samples


def dpcm_decode(samples: list[int]) -> list[int]:
    """
    Decodes DPCM encoded samples
    """

    decoded_samples = []

    accumulator = 0
    for quantized_diff in samples:
        # calculate difference
        diff = DPCM_MAP[quantized_diff]

        # add to accumulator
        accumulator = max(min(accumulator + diff, 127), -128)

        # add to decoded samples
        decoded_samples.append(int(accumulator) ^ 0b0111_1111)

    return decoded_samples


def read_wav_file(file_path):
    """
    Reads .wav file
    """

    with wave.open(file_path, 'rb') as wav_file:
        parameters = wav_file.getparams()
        samples = wav_file.readframes(parameters.nframes)

    return parameters, samples


def write_wav_file(file_path, parameters, processed_frames):
    """
    Write .wav file
    """

    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setparams(parameters)
        wav_file.writeframes(processed_frames)


def process_audio_samples(parameters, frames):
    """
    Raw file creation
    """

    if parameters.sampwidth == 1:
        byte_width = "b"
    elif parameters.sampwidth == 2:
        byte_width = "h"
    else:
        raise NotImplementedError

    fmt = "<" + byte_width * parameters.nframes * parameters.nchannels

    # unpack samples
    samples = list(struct.unpack(fmt, frames))

    # process
    encoded_samples = dpcm_encode(samples)
    decoded_samples = dpcm_decode(encoded_samples)

    fmt = "<" + byte_width * len(decoded_samples) * parameters.nchannels

    # pack
    return struct.pack(fmt, *decoded_samples)


def encode_wav(input_file: str, output_file: str) -> None:
    """
    Encodes a .wav file
    """


def decode_wav(input_file: str, output_file: str) -> None:
    """
    Decodes a .wav DPCM encoded file
    """


def main():
    # parse arguments
    parser = argparse.ArgumentParser(prog="Differential PCM codec")

    # add arguments
    parser.add_argument(
        "-i", "--input",
        help="file input",
        required=True)
    parser.add_argument(
        "-o", "--output",
        help="file output")
    parser.add_argument(
        "--mode",
        help="modes of DPCM codec",
        choices=["encode_wav", "decode_wav"],
        required=True)

    # parse arguments
    args = parser.parse_args()

    # make file names
    input_file = args.input
    output_file = args.output if args.output else "out_" + args.input

    if args.mode == "encode_wav":
        encode_wav(input_file, output_file)
    elif args.mode == "decode_wav":
        decode_wav(input_file, output_file)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()

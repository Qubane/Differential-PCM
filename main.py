import os
import wave
import struct
import argparse


DPCM_SIZE = 16


def _dpcm_mapping(x):
    """
    Generates DPCM mapping
    """

    # mapping type
    value = int(abs(x) ** 2) + 1

    # keep the sign
    if x < 0:
        return -value
    return value


# DPCM mapping
DPCM_MAP = [_dpcm_mapping(x + 0.5) for x in range(-DPCM_SIZE // 2, DPCM_SIZE // 2)]


def dpcm_quantize(value: int) -> int:
    """
    Quantize the value
    """

    # for positive values
    if value >= 0:
        for idx in range(DPCM_SIZE // 2, DPCM_SIZE):
            if value - DPCM_MAP[idx] <= 0:
                return idx
        return DPCM_SIZE - 1

    # for negative values
    else:
        for idx in range(DPCM_SIZE // 2 - 1, -1, -1):
            if value - DPCM_MAP[idx] >= 0:
                return idx
        return 0


def dpcm_encode(samples: list[int], sample_width: int = 1, signed: bool = True) -> list[int]:
    """
    Encodes samples using DPCM
    :param samples: list of integer samples
    :param sample_width: byte-width of each sample
    :param signed: are the samples signed integers
    :return: 4 bit DPCM compressed samples
    """

    # encoded samples
    encoded_samples = []

    # make signed correction mask
    if not signed:
        correction_mask = 0
    else:
        correction_mask = (2 ** (8 * sample_width) - 1) >> 1

    # perform DPCM
    accumulator = 0
    for sample in samples:
        # calculate difference
        diff = (sample ^ correction_mask) - accumulator

        # map to range
        quantized_diff = dpcm_quantize(diff)

        # append to encoded samples
        encoded_samples.append(quantized_diff)

        accumulator += DPCM_MAP[quantized_diff]

    # return encoded samples
    return encoded_samples


def dpcm_decode(samples: list[int], sample_width: int = 1, signed: bool = True) -> list[int]:
    """
    Decodes DPCM encoded samples
    :param samples: list of DPCM encoded samples
    :param sample_width: byte-width of each sample
    :param signed: are the samples signed integers
    :return: decompressed samples
    """

    # decoded samples
    decoded_samples = []

    # make signed correction mask
    if not signed:
        correction_mask = 0
    else:
        correction_mask = (2 ** (8 * sample_width) - 1) >> 1

    # make integer start and stop ranges
    int_start = correction_mask if signed else 0
    int_stop = -(correction_mask + 1) if signed else ((correction_mask << 1) + 1)

    # perform DPCM decoding
    accumulator = 0
    for quantized_diff in samples:
        # calculate difference
        diff = DPCM_MAP[quantized_diff]

        # add to accumulator
        accumulator = max(min(accumulator + diff, int_start), int_stop)

        # add to decoded samples
        decoded_samples.append(int(accumulator) ^ correction_mask)

    # return decoded samples
    return decoded_samples


def make_wave_parameters(parameters) -> dict[str, int]:
    """
    Makes parameters from "wave._wave_params" class that is not type hintable
    """

    # return essential parameters
    return {
        "sampwidth": parameters.sampwidth,
        "nchannels": parameters.nchannels,
        "framerate": parameters.framerate,
        "nframes": parameters.nframes}


def read_wav_file(file_path: str) -> tuple[bytes, dict[str, int]]:
    """
    Reads .wav file
    :param file_path: path to file
    """

    # read file
    with wave.open(file_path, 'rb') as wav_file:
        parameters = make_wave_parameters(wav_file.getparams())
        frames = wav_file.readframes(parameters["nframes"])

    # output data
    return frames, parameters


def write_wav_file(file_path: str, frames: bytes, parameters: dict[str, int]):
    """
    Write .wav file
    :param file_path: path to file
    :param frames: frames to write
    :param parameters: wave file parameters
    """

    # write file
    with wave.open(file_path, 'wb') as wav_file:
        # write parameters
        wav_file.setsampwidth(parameters["sampwidth"])
        wav_file.setnchannels(parameters["nchannels"])
        wav_file.setframerate(parameters["framerate"])

        # write frames
        wav_file.writeframes(frames)


def unpack_frames(frames: bytes, parameters) -> list[int]:
    """
    Unpacks the raw frame bytes
    """

    # figure out byte width
    if parameters.sampwidth == 1:
        byte_width = "b"
    elif parameters.sampwidth == 2:
        byte_width = "h"
    else:
        raise NotImplementedError

    # generate fmt
    fmt = "<" + byte_width * parameters.nframes * parameters.nchannels

    # return unpacked samples
    return list(struct.unpack(fmt, frames))


def pack_frames(samples: list[int], parameters) -> bytes:
    """
    Packs samples back into raw frames
    """

    # make sure parameters is a dict
    if not isinstance(parameters, dict):
        parameters = {
            "sampwidth": parameters.sampwidth,
            "nchannels": parameters.nchannels}

    # figure out byte width
    if parameters["sampwidth"] == 1:
        byte_width = "b"
    elif parameters["sampwidth"] == 2:
        byte_width = "h"
    else:
        raise NotImplementedError

    # generate fmt
    fmt = "<" + byte_width * len(samples) * parameters["nchannels"]

    # return packed samples
    return struct.pack(fmt, *samples)


def pack_dpcm(samples: list[int], parameters) -> bytes:
    """
    Packs DPCM binary
    """

    # file format
    # byte width
    # channel number
    # framerate
    # [samples]
    fmt = "<BBL"
    fmt += "B" * (len(samples) // 2)

    # pack samples
    packed_samples = []
    for idx in range(0, len(samples) - 1, 2):
        packed_samples.append((samples[idx] << 4) + samples[idx + 1])

    # return packed
    return struct.pack(
        fmt,
        parameters.sampwidth,
        parameters.nchannels,
        parameters.framerate,
        *packed_samples)


def unpack_dpcm(packed: bytes) -> tuple[list[int], dict[str, int]]:
    """
    Unpacks DPCM packed bytes
    """

    # file format
    # sample width
    # channel number
    # [samples]
    fmt = "<BBL"
    fmt += "B" * (len(packed) - 2 - 4)

    # unpack raw
    unpacked = struct.unpack(fmt, packed)

    # make parameters
    parameters = {
        "sampwidth": unpacked[0],
        "nchannels": unpacked[1],
        "framerate": unpacked[2]}

    # unpack samples
    unpacked_samples = []
    for packed_sample in unpacked[3:]:
        unpacked_samples.append(packed_sample >> 4)
        unpacked_samples.append(packed_sample & 0xF)

    # return unpacked
    return unpacked_samples, parameters


def encode_wav(input_file: str, output_file: str) -> None:
    """
    Encodes a .wav file
    """

    # read file
    parameters, frames = read_wav_file(input_file)

    # unpack samples
    samples = unpack_frames(frames, parameters)

    # encode samples
    samples = dpcm_encode(samples)

    # pack bytes
    packed_dpcm = pack_dpcm(samples, parameters)

    # store into file
    with open(output_file, "wb") as file:
        file.write(packed_dpcm)


def decode_wav(input_file: str, output_file: str) -> None:
    """
    Decodes a .wav DPCM encoded file
    """

    # read file
    with open(input_file, "rb") as file:
        packed_dpcm = file.read()

    # unpack DPCM
    samples, parameters = unpack_dpcm(packed_dpcm)

    # decode samples
    samples = dpcm_decode(samples)

    # convert into frames
    frames = pack_frames(samples, parameters)

    # store into file
    write_wav_file(output_file, parameters, frames)


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
        output_file = os.path.splitext(output_file)[0] + ".dpcm"
        encode_wav(input_file, output_file)
    elif args.mode == "decode_wav":
        decode_wav(input_file, output_file)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()

import os
import wave
import struct
import argparse


class DPCMCompressor:
    """
    Differential PCM compressor class
    """

    def __init__(self, dpcm_depth: int):
        self.dpcm_depth: int = dpcm_depth
        self.dpcm_size: int = 2 ** dpcm_depth

        self.difference_mapping: list[float] = []
        self._make_mapping()

    def _make_mapping(self):
        """
        Generates a difference mapping for compressor
        """

        # functions were picked based on "good vibes"
        if self.dpcm_depth == 1:
            diff_function = lambda x: abs(x) * 10
        elif self.dpcm_depth == 2:
            diff_function = lambda x: abs(x) ** 4 + 4
        elif self.dpcm_depth == 4:
            diff_function = lambda x: abs(x) ** 2 + 1
        else:
            raise NotImplementedError

        # generate difference mapping
        self.difference_mapping = [diff_function(x + 0.5) for x in range(-self.dpcm_size // 2, self.dpcm_size // 2)]

    def quantize(self, value: int) -> int:
        """
        Quantizes the value using the difference mapping
        :param value: value to quantize
        :return: integer index of quantized mapping
        """

        if value >= 0:
            for idx in range(self.dpcm_size // 2, self.dpcm_size):
                if value - self.difference_mapping[idx] <= 0:
                    return idx
            return self.dpcm_size - 1
        else:
            for idx in range(self.dpcm_size // 2 - 1, -1, -1):
                if value - self.difference_mapping[idx] >= 0:
                    return idx
            return 0

    def encode(self, samples: list[int], sample_width: int = 1) -> list[int]:
        """
        Encodes samples using DPCM
        :param samples: list of integer samples
        :param sample_width: byte-width of each sample
        :return: 4 bit DPCM compressed samples
        """

        # encoded samples
        encoded_samples = []

        # make sample width correction
        sample_width_offset = 8 if sample_width == 2 else 0

        # make signed correction mask
        correction_mask = 127 if sample_width == 1 else 0

        # perform DPCM
        accumulator = 0
        for sample in samples:
            # calculate difference
            diff = ((sample >> sample_width_offset) ^ correction_mask) - accumulator

            # map to range
            quantized_diff = self.quantize(diff)

            # append to encoded samples
            encoded_samples.append(quantized_diff)

            accumulator += self.difference_mapping[quantized_diff]

        # return encoded samples
        return encoded_samples

    def dpcm_decode(self, samples: list[int], sample_width: int = 1) -> list[int]:
        """
        Decodes DPCM encoded samples
        :param samples: list of DPCM encoded samples
        :param sample_width: byte-width of each sample
        :return: decompressed samples
        """

        # decoded samples
        decoded_samples = []

        # make sample width correction
        sample_width_offset = 8 if sample_width == 2 else 0

        # make signed correction mask
        correction_mask = 127 if sample_width == 1 else 0

        # perform DPCM decoding
        accumulator = 0
        for quantized_diff in samples:
            # calculate difference
            diff = self.difference_mapping[quantized_diff]

            # add to accumulator
            accumulator = max(min(accumulator + diff, 127), -127)

            # add to decoded samples
            decoded_samples.append((int(accumulator) ^ correction_mask) << sample_width_offset)

        # return decoded samples
        return decoded_samples

    def pack_dpcm(self, samples: list[int], parameters: dict[str, int]) -> bytes:
        """
        Packs DPCM encoded samples into bytes
        :param samples: dpcm encoded samples
        :param parameters: sample parameters
        """

        # file format
        # byte width
        # channel number
        # framerate
        # [samples]
        fmt = "<BBL"

        offset = 8 // self.dpcm_size
        fmt += "B" * (len(samples) // offset)

        # pack samples
        packed_samples = []
        for idx in range(0, len(samples) - 1, offset):
            if self.dpcm_size == 1:
                acc = sum([samples[idx + x] << (offset - x - 1) for x in range(offset)])
                packed_samples.append(acc)
            elif self.dpcm_size == 2:
                packed_samples.append(
                    (samples[idx] << 6) + (samples[idx + 1] << 4) + (samples[idx + 2] << 2) + samples[idx + 3])
            elif self.dpcm_size == 4:
                packed_samples.append((samples[idx] << 4) + samples[idx + 1])
            else:
                raise NotImplementedError

        # return packed
        return struct.pack(
            fmt,
            parameters["sampwidth"],
            parameters["nchannels"],
            parameters["framerate"],
            *packed_samples)

    def unpack_dpcm(self, packed: bytes) -> tuple[list[int], dict[str, int]]:
        """
        Unpacks DPCM packed bytes
        :param packed: packed DPCM compressed data
        :return: tuple of samples and sample parameters
        """

        # file format
        # byte width
        # channel number
        # framerate
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
            if self.dpcm_size == 1:
                unpacked_samples.append(packed_sample >> 7)
                unpacked_samples.append((packed_sample >> 6) & 1)
                unpacked_samples.append((packed_sample >> 5) & 1)
                unpacked_samples.append((packed_sample >> 4) & 1)
                unpacked_samples.append((packed_sample >> 3) & 1)
                unpacked_samples.append((packed_sample >> 2) & 1)
                unpacked_samples.append((packed_sample >> 1) & 1)
                unpacked_samples.append(packed_sample & 1)
            elif self.dpcm_size == 2:
                unpacked_samples.append(packed_sample >> 6)
                unpacked_samples.append((packed_sample >> 4) & 0b11)
                unpacked_samples.append((packed_sample >> 2) & 0b11)
                unpacked_samples.append(packed_sample & 0b11)
            elif self.dpcm_size == 4:
                unpacked_samples.append(packed_sample >> 4)
                unpacked_samples.append(packed_sample & 0xF)
            else:
                raise NotImplementedError

        # return unpacked
        return unpacked_samples, parameters


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


def unpack_frames(frames: bytes, parameters: dict[str, int]) -> list[int]:
    """
    Unpacks the raw frame bytes
    :param frames: frames to unpack
    :param parameters: sample parameters
    """

    # figure out byte width
    if parameters["sampwidth"] == 1:
        byte_width = "b"
    elif parameters["sampwidth"] == 2:
        byte_width = "h"
    else:
        raise NotImplementedError

    # generate fmt
    fmt = "<" + byte_width * parameters["nframes"] * parameters["nchannels"]

    # return unpacked samples
    return list(struct.unpack(fmt, frames))


def pack_frames(samples: list[int], parameters: dict[str, int]) -> bytes:
    """
    Packs samples back into raw frames
    :param samples: samples to pack
    :param parameters: sample parameters
    """

    # figure out byte width
    if parameters["sampwidth"] == 1:
        byte_width = "b"
    elif parameters["sampwidth"] == 2:
        byte_width = "h"
    else:
        raise NotImplementedError

    # generate fmt
    fmt = "<" + byte_width * len(samples)

    # return packed_dpcm samples
    return struct.pack(fmt, *samples)


def pack_dpcm(samples: list[int], parameters: dict[str, int]) -> bytes:
    """
    Packs DPCM binary
    :param samples: samples to pack
    :param parameters: sample parameters
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

    # return packed_dpcm
    return struct.pack(
        fmt,
        parameters["sampwidth"],
        parameters["nchannels"],
        parameters["framerate"],
        *packed_samples)


def unpack_dpcm(packed_dpcm: bytes) -> tuple[list[int], dict[str, int]]:
    """
    Unpacks DPCM packed bytes
    :param packed_dpcm: packed DPCM compressed data
    :return: tuple of samples and sample parameters
    """

    # file format
    # sample width
    # channel number
    # [samples]
    fmt = "<BBL"
    fmt += "B" * (len(packed_dpcm) - 2 - 4)

    # unpack raw
    unpacked = struct.unpack(fmt, packed_dpcm)

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


def split_tracks(samples: list[int], nchannels: int) -> list[list[int]]:
    """
    Splits single samples list into multiple tracks
    :param samples: list of samples
    :param nchannels: number of channels
    """

    return [samples[offset::nchannels] for offset in range(nchannels)]


def merge_tracks(samples: list[int], tracks: list[list[int]], nchannels: int):
    """
    Splits single samples list into multiple tracks
    :param samples: original samples list
    :param tracks: list of tracks
    :param nchannels: number of channels
    """

    for offset in range(nchannels):
        samples[offset::nchannels] = tracks[offset]


def encode_wav(input_file: str, output_file: str) -> None:
    """
    Encodes a .wav file
    :param input_file: input .wav file
    :param output_file: output .dpcm file
    """

    # read file
    frames, parameters = read_wav_file(input_file)

    # unpack samples
    samples = unpack_frames(frames, parameters)

    # split tracks
    tracks = split_tracks(samples, parameters["nchannels"])

    # encode tracks
    for track_idx in range(parameters["nchannels"]):
        tracks[track_idx] = dpcm_encode(tracks[track_idx], sample_width=parameters["sampwidth"])

    # merge multiple tracks
    merge_tracks(samples, tracks, parameters["nchannels"])

    # pack bytes
    packed_dpcm = pack_dpcm(samples, parameters)

    # store into file
    with open(output_file, "wb") as file:
        file.write(packed_dpcm)


def decode_wav(input_file: str, output_file: str) -> None:
    """
    Decodes a .wav DPCM encoded file
    :param input_file: input .dpcm file
    :param output_file: output .wav file
    """

    # read file
    with open(input_file, "rb") as file:
        packed_dpcm = file.read()

    # unpack DPCM
    samples, parameters = unpack_dpcm(packed_dpcm)

    # split tracks
    tracks = split_tracks(samples, parameters["nchannels"])

    # decode tracks
    for track_idx in range(parameters["nchannels"]):
        tracks[track_idx] = dpcm_decode(tracks[track_idx], sample_width=parameters["sampwidth"])

    # merge multiple tracks
    merge_tracks(samples, tracks, parameters["nchannels"])

    # convert into frames
    frames = pack_frames(samples, parameters)

    # store into file
    write_wav_file(output_file, frames, parameters)


def squeeze(input_file: str, output_file: str) -> None:
    """
    Compresses and then decompresses the file, essentially just making quality worse
    :param input_file: input .wav file
    :param output_file: output .wav file
    """

    # read file
    frames, parameters = read_wav_file(input_file)

    # unpack samples
    samples = unpack_frames(frames, parameters)

    # split tracks
    tracks = split_tracks(samples, parameters["nchannels"])

    # encode & decode tracks (squeeze)
    for track_idx in range(parameters["nchannels"]):
        tracks[track_idx] = dpcm_encode(tracks[track_idx], sample_width=parameters["sampwidth"])
        tracks[track_idx] = dpcm_decode(tracks[track_idx], sample_width=parameters["sampwidth"])

    # merge multiple tracks
    merge_tracks(samples, tracks, parameters["nchannels"])

    # convert into frames
    frames = pack_frames(samples, parameters)

    # store into file
    write_wav_file(output_file, frames, parameters)


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
        choices=["encode_wav", "decode_wav", "squeeze"],
        required=True)
    parser.add_argument(
        "--dpcm-depth",
        help="DPCM bit depth (1 - least quality & most compression)",
        choices=[1, 2, 4],
        default=4)

    # parse arguments
    args = parser.parse_args()

    # make file names
    input_file = args.input
    output_file = args.output if args.output else "out_" + args.input

    # pick mode
    if args.mode == "encode_wav":
        output_file = os.path.splitext(output_file)[0] + ".dpcm"
        encode_wav(input_file, output_file)
    elif args.mode == "decode_wav":
        decode_wav(input_file, output_file)
    elif args.mode == "squeeze":
        squeeze(input_file, output_file)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()

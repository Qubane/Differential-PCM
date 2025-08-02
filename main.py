import os
import wave
import struct
import argparse
import numpy as np


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


class DPCMCompressor:
    """
    Differential PCM compressor class
    """

    def __init__(self, dpcm_depth: int):
        self.dpcm_depth: int = 0
        self.dpcm_size: int = 0

        # difference mapping
        self.difference_mapping: np.ndarray | None = None

        # update dpcm depth
        self._set_dpcm(dpcm_depth)

    def _set_dpcm(self, depth: int):
        """
        Sets own DPCM depth
        """

        self.dpcm_depth = depth
        self.dpcm_size = 2 ** self.dpcm_depth

        self._make_mapping()

    def _make_mapping(self):
        """
        Generates a difference mapping for compressor
        """

        # functions were picked based on "good vibes"
        if self.dpcm_depth == 1:
            diff_function = lambda x: abs(x) * 16
        elif self.dpcm_depth == 2:
            diff_function = lambda x: (abs(x) ** 4 + 4) * np.sign(x)
        elif self.dpcm_depth == 4:
            diff_function = lambda x: (abs(x) ** 2 + 1) * np.sign(x)
        else:
            raise NotImplementedError

        # generate difference mapping
        self.difference_mapping = np.arange(-self.dpcm_size / 2, self.dpcm_size / 2) + 0.5
        self.difference_mapping = np.vectorize(diff_function)(self.difference_mapping)

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

    def encode(self, samples: np.ndarray) -> np.ndarray:
        """
        Encodes samples using DPCM
        :param samples: integer samples
        :return: list of unquantized differences
        """

        # encoded samples
        encoded_samples = np.zeros(samples.shape, dtype=np.int32)

        # perform DPCM
        accumulator = 0
        for idx, sample in enumerate(samples):
            # calculate difference
            diff = sample - accumulator

            # append to encoded samples
            encoded_samples[idx] = diff

            # update accumulator
            accumulator += diff

        # return encoded samples
        return encoded_samples

    def decode(self, samples: list[int], sample_width: int = 1) -> list[int]:
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

    def _packing_format(self, sample_count: int) -> str:
        """
        Generates a packing / unpacking format string for DPCM compressed files
        """

        # file format (little-endian)
        fmt = "<"

        # byte width (1 byte unsigned integer)
        fmt += "B"

        # channel number (1 byte unsigned integer)
        fmt += "B"

        # framerate (4 byte unsigned integer)
        fmt += "L"

        # DPCM depth (1 byte unsigned integer)
        fmt += "B"

        # [samples] (1 byte unsigned integer array)
        fmt += "B" * sample_count

        # return format
        return fmt

    def pack_dpcm(self, samples: list[int], parameters: dict[str, int]) -> bytes:
        """
        Packs DPCM encoded samples into bytes
        :param samples: dpcm encoded samples
        :param parameters: sample parameters
        """

        # calculate offset
        offset = 8 // self.dpcm_depth

        # make format
        fmt = self._packing_format(len(samples) // offset)

        # pack samples
        packed_samples = []
        for idx in range(0, len(samples) - (offset - 1), offset):
            if self.dpcm_depth == 1:
                acc = sum([samples[idx + x] << (offset - x - 1) for x in range(offset)])
                packed_samples.append(acc)
            elif self.dpcm_depth == 2:
                packed_samples.append(
                    (samples[idx] << 6) + (samples[idx + 1] << 4) + (samples[idx + 2] << 2) + samples[idx + 3])
            elif self.dpcm_depth == 4:
                packed_samples.append((samples[idx] << 4) + samples[idx + 1])
            else:
                raise NotImplementedError

        # return packed
        return struct.pack(
            fmt,
            parameters["sampwidth"],
            parameters["nchannels"],
            parameters["framerate"],
            self.dpcm_depth,
            *packed_samples)

    def unpack_dpcm(self, packed: bytes) -> tuple[list[int], dict[str, int]]:
        """
        Unpacks DPCM packed bytes
        :param packed: packed DPCM compressed data
        :return: tuple of samples and sample parameters
        """

        # file format
        # magic number refers to sample array start in bytes
        fmt = self._packing_format(len(packed) - 3 - 4)

        # unpack raw
        unpacked = struct.unpack(fmt, packed)

        # make parameters
        parameters = {
            "sampwidth": unpacked[0],
            "nchannels": unpacked[1],
            "framerate": unpacked[2]}

        # set dpcm depth
        self._set_dpcm(unpacked[3])

        # unpack samples
        unpacked_samples = []
        for packed_sample in unpacked[4:]:
            if self.dpcm_depth == 1:
                unpacked_samples.append(packed_sample >> 7)
                unpacked_samples.append((packed_sample >> 6) & 1)
                unpacked_samples.append((packed_sample >> 5) & 1)
                unpacked_samples.append((packed_sample >> 4) & 1)
                unpacked_samples.append((packed_sample >> 3) & 1)
                unpacked_samples.append((packed_sample >> 2) & 1)
                unpacked_samples.append((packed_sample >> 1) & 1)
                unpacked_samples.append(packed_sample & 1)
            elif self.dpcm_depth == 2:
                unpacked_samples.append(packed_sample >> 6)
                unpacked_samples.append((packed_sample >> 4) & 0b11)
                unpacked_samples.append((packed_sample >> 2) & 0b11)
                unpacked_samples.append(packed_sample & 0b11)
            elif self.dpcm_depth == 4:
                unpacked_samples.append(packed_sample >> 4)
                unpacked_samples.append(packed_sample & 0xF)
            else:
                raise NotImplementedError

        # return unpacked
        return unpacked_samples, parameters


class Application:
    """
    Application class
    """

    def __init__(self):
        self.parser_input_file: str = ""
        self.parser_output_file: str = ""
        self.parser_dpcm_depth: int = 0
        self.parser_mode: str = ""

        self.compressor: DPCMCompressor | None = None

    def parser_args(self):
        """
        Parses CLI arguments
        """

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
            choices=["1", "2", "4"],
            default="4")

        # parse arguments
        args = parser.parse_args()

        # set parser arguments
        self.parser_input_file = args.input
        self.parser_output_file = args.output if args.output else "out_" + args.input
        self.parser_dpcm_depth = int(args.dpcm_depth)
        self.parser_mode = args.mode

    def run(self):
        """
        Runs the application
        """

        # parse args
        self.parser_args()

        # compressor
        self.compressor = DPCMCompressor(dpcm_depth=self.parser_dpcm_depth)

        # pick mode
        if self.parser_mode == "encode_wav":
            self.parser_output_file = os.path.splitext(self.parser_output_file)[0] + ".dpcm"
            self.encode_wav()
        elif self.parser_mode == "decode_wav":
            self.decode_wav()
        elif self.parser_mode == "squeeze":
            self.squeeze()
        else:
            raise NotImplementedError

    def encode_wav(self) -> None:
        """
        Encodes a .wav file
        """

        # read file
        frames, parameters = read_wav_file(self.parser_input_file)

        # unpack samples
        samples = unpack_frames(frames, parameters)

        # split tracks
        tracks = split_tracks(samples, parameters["nchannels"])

        # encode tracks
        for track_idx in range(parameters["nchannels"]):
            tracks[track_idx] = self.compressor.encode(tracks[track_idx], sample_width=parameters["sampwidth"])

        # merge multiple tracks
        merge_tracks(samples, tracks, parameters["nchannels"])

        # pack bytes
        packed_dpcm = self.compressor.pack_dpcm(samples, parameters)

        # store into file
        with open(self.parser_output_file, "wb") as file:
            file.write(packed_dpcm)

    def decode_wav(self) -> None:
        """
        Decodes a .wav DPCM encoded file
        """

        # read file
        with open(self.parser_input_file, "rb") as file:
            packed_dpcm = file.read()

        # unpack DPCM
        samples, parameters = self.compressor.unpack_dpcm(packed_dpcm)

        # split tracks
        tracks = split_tracks(samples, parameters["nchannels"])

        # decode tracks
        for track_idx in range(parameters["nchannels"]):
            tracks[track_idx] = self.compressor.decode(tracks[track_idx], sample_width=parameters["sampwidth"])

        # merge multiple tracks
        merge_tracks(samples, tracks, parameters["nchannels"])

        # convert into frames
        frames = pack_frames(samples, parameters)

        # store into file
        write_wav_file(self.parser_output_file, frames, parameters)

    def squeeze(self) -> None:
        """
        Compresses and then decompresses the file, essentially just making quality worse
        """

        # read file
        frames, parameters = read_wav_file(self.parser_input_file)

        # unpack samples
        samples = unpack_frames(frames, parameters)

        # split tracks
        tracks = split_tracks(samples, parameters["nchannels"])

        # encode & decode tracks (squeeze)
        for track_idx in range(parameters["nchannels"]):
            tracks[track_idx] = self.compressor.encode(tracks[track_idx], sample_width=parameters["sampwidth"])
            tracks[track_idx] = self.compressor.decode(tracks[track_idx], sample_width=parameters["sampwidth"])

        # merge multiple tracks
        merge_tracks(samples, tracks, parameters["nchannels"])

        # convert into frames
        frames = pack_frames(samples, parameters)

        # store into file
        write_wav_file(self.parser_output_file, frames, parameters)


def main():
    # app = Application()
    # app.run()

    import matplotlib.pyplot as plt

    plot_start = 0
    plot_end = plot_start + 128

    compressor = DPCMCompressor(4)

    # read file
    frames, parameters = read_wav_file("tests/sine_8.wav")

    # unpack samples
    samples = np.array(unpack_frames(frames, parameters), dtype=np.int16).astype(np.int32)

    # idk why signed byte integers need that
    if parameters["sampwidth"] == 1:
        samples ^= 127

    plt.subplot(3, 1, 1)
    plt.plot(samples[plot_start:plot_end])

    samples = compressor.encode(samples)

    plt.subplot(3, 1, 2)
    plt.plot(samples[plot_start:plot_end])

    samples = compressor.encode(samples)

    plt.subplot(3, 1, 3)
    plt.plot(samples[plot_start:plot_end])

    plt.show()


if __name__ == '__main__':
    main()

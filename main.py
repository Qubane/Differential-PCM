import wave
import cmath
import struct
import matplotlib.pyplot as plt


DPCM_SIZE = 16


def _dpcm_mapping(x):
    value = int(abs(x) * 6) + 1
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
                return idx
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

    # range_start = 0
    # range_stop = range_start + 512
    #
    # plt.subplot(3, 1, 1)
    # plt.plot(samples[range_start:range_stop])
    #
    # plt.subplot(3, 1, 2)
    # plt.plot(encoded_samples[range_start:range_stop])
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(decoded_samples[range_start:range_stop])
    #
    # plt.show()

    fmt = "<" + byte_width * len(decoded_samples) * parameters.nchannels

    # pack
    return struct.pack(fmt, *decoded_samples)


def main():
    input_file_path = 'tests/test0__orig.wav'
    output_file_path = 'output.wav'

    parameters, frames = read_wav_file(input_file_path)

    processed_frames = process_audio_samples(parameters, frames)
    write_wav_file(output_file_path, parameters, processed_frames)


if __name__ == '__main__':
    main()

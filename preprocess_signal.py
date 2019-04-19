import librosa
import librosa.display
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# ---- OPTIONS TAMPLATE ----- #
#    dic = {'FS': 22050,
#           'N_FFT': 2048,
#           'WIN': 'hamming',
#           'HOP_t': 0.014,
#           'WIN_t': 0.046,
#           # MEL
#           'N_MEL': 80,
#           'F_MIN': 50,
#           'F_MAX': 12000,
#           'expected_len': 700,
#           }
# --------------------------- #


def define_param(option):
    dic = {'FS': 22050,
           'N_FFT': 4096,
           'WIN': 'hamming',
           'HOP_t': 0.014,
           'WIN_t': 0.046,
           # MEL
           'N_MEL': 80,
           'F_MIN': 50,
           'F_MAX': 12000,
           'expected_len': 700,
           }
    if option == 'baseline':
        return dic
    elif option == 'temporal':
        dic['WIN_t'] = 0.012
        dic['HOP_t'] = 0.006
        dic['expected_len'] = 1669
        return dic
    elif option == 'frequential':
        dic['WIN_t'] = 0.032
        dic['HOP_t'] = 0.016
        dic['N_MEL'] = 160
        dic['expected_len'] = 624
        return dic


def compute_spectrogram(path, options):
    x, fs = librosa.load(path)
    # Resample
    if fs != options['FS']:
        x = librosa.resample(x, fs, options['FS'])
        fs = options['FS']
    # Compute sfft
    sfft_spec = librosa.core.stft(x, n_fft=options['N_FFT'],
                                  hop_length=int(options['HOP_t']*fs),
                                  win_length=int(options['WIN_t']*fs),
                                  window=options['WIN'])
    # Log Spectogram
    lfeat = librosa.core.power_to_db(np.abs(sfft_spec)**2)
    # Convert to an array
    lfeat = np.array(lfeat)

    return lfeat.transpose()


def compute_spectrogram_mel(path, options):
    x, fs = librosa.load(path)
    # Resample
    if fs != options['FS']:
        x = librosa.resample(x, fs, options['FS'])
        fs = options['FS']
    # Compute sfft
    # N_FFT = int(len(x) / 2)
    sfft_spec = librosa.core.stft(x, n_fft=options['N_FFT'],
                                  hop_length=int(options['HOP_t']*fs),
                                  win_length=int(options['WIN_t']*fs),
                                  window=options['WIN'])
    # Create Mel filter
    mel_filter = librosa.filters.mel(fs, n_fft=options['N_FFT'],
                                     n_mels=options['N_MEL'],
                                     fmin=options['F_MIN'],
                                     fmax=options['F_MAX'])
    # Filtering
    mel_feat = np.dot(mel_filter, sfft_spec)
    # Log Mel Spectrogram
    lmel_feat = librosa.core.power_to_db(np.abs(mel_feat)**2)
    # Convert to an array
    lmel_feat = np.array(lmel_feat)

    return lmel_feat.transpose()


def normalization(data):
    # Normalize array
    max_value = np.amax(data)
    min_value = np.amin(data)
    norm_data = (data - min_value)/(max_value - min_value)

    return norm_data


def save_spectogram(data, path, name):
    np.save(os.path.join(path, name), data)


def plot_spectogram(data, options):
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(data, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()


def plot_spectogram_mel(data, options):
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(data, sr=options['FS'],
                             hop_length=options['HOP_t']*options['FS'],
                             fmin=options['F_MIN'], fmax=options['F_MAX'],
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Pre processing Signal Tool")
    parser.add_argument('input_file', help='Path with the .vaw files')
    parser.add_argument('output_file', help='Path to save the spectograms')
    parser.add_argument('--type', choices=['normal', 'mel'],
                        help='Compute with Mel Coef or simple spectogram')
    parser.add_argument('--process', choices=['baseline', 'temporal',
                                              'frequential'],
                        help='Choose type of process signal')
    parser.add_argument('--norm', choices=['none', 'individual', 'full'],
                        help='Choose the normalization of the signal')
    args = parser.parse_args()

    options = define_param(args.process)
    max_value = 0
    min_value = 0
    for wave in os.listdir(args.input_file):
        file_path = os.path.join(args.input_file, wave)
        if args.type == 'normal':
            features = compute_spectrogram(file_path, options)
        elif args.type == 'mel':
            features = compute_spectrogram_mel(file_path, options)
        # Normalization
        if args.norm == 'individual':
            features = normalization(features)

        elif args.norm == 'full':
            if np.amax(features) > max_value:
                max_value = np.amax(features)
            if np.amin(features) < min_value:
                min_value = np.amin(features)

        if len(features) > options['expected_len']:
            features = np.resize(features, (options['expected_len'],
                                            options['N_MEL']))
        # elif len(features) < expected_len:
        # TODO: Complete
        save_spectogram(features, args.output_file, wave)

    print('Max Value: {}'.format(max_value))
    print('Min Value: {}'.format(min_value))

import librosa
import librosa.display
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def define_param(option):
    dic = {'N_FFT': 2048,
           'WIN': 'hamming',
           'HOP_L': 0.014,
           'WIN_L': 0.046,
           # MEL
           'N_MEL': 80,
           'F_MIN': 50,
           'F_MAX': 12000,
           'expected_len': 700,
           }
    if option == 'baseline':
        return dic
    elif option == 'temporal':
        return dic
    elif option == 'frequential':
        return dic


def compute_spectrogram(path, options):
    x, fs = librosa.load(path)
    # Resample
    if fs != 22050:
        x = librosa.resample(x, fs, 22050)
        fs = 22050
    # Compute sfft
    sfft_spec = librosa.core.stft(x, n_fft=options['N_FFT'],
                                  hop_length=int(options['HOP_L']*fs),
                                  win_length=int(options['WIN_L']*fs),
                                  window=options['WIN'])
    # Log Spectogram
    lfeat = librosa.core.power_to_db(np.abs(sfft_spec)**2)
    # Convert to an array
    lfeat = np.array(lfeat)
    # Normalize array
    max_value = np.amax(lfeat)
    min_value = np.amin(lfeat)
    lfeat_n = (lfeat - min_value)/(max_value - min_value)

    return lfeat_n.transpose()


def compute_spectrogram_mel(path, options):
    x, fs = librosa.load(path)
    # Resample
    if fs != 22050:
        x = librosa.resample(x, fs, 22050)
        fs = 22050
    # Compute sfft
    # N_FFT = int(len(x) / 2)
    sfft_spec = librosa.core.stft(x, n_fft=options['N_FFT'],
                                  hop_length=int(options['HOP_L']*fs),
                                  win_length=int(options['WIN_L']*fs),
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
    # Normalize array
    max_value = np.amax(lmel_feat)
    min_value = np.amin(lmel_feat)
    lmel_feat_n = (lmel_feat - min_value)/(max_value - min_value)

    return lmel_feat_n.transpose()


def save_spectogram(data, path, name):
    np.save(os.path.join(path, name), data)


def plot_spectogram(data, options):
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(data, fmax=options['F_MAX'], x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()


def plot_spectogram_mel(data, options):
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(data, y_axis='mel', fmax=options['F_MAX'],
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
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
    args = parser.parse_args()

    options = define_param(args.process)
    for wave in os.listdir(args.input_file):
        file_path = os.path.join(args.input_file, wave)
        if args.type == 'normal':
            features = compute_spectrogram(file_path, options)
        elif args.type == 'mel':
            features = compute_spectrogram_mel(file_path, options)

        if len(features) > options['expected_len']:
            features = np.resize(features, (700, 80))
        # elif len(features) < expected_len:
        # TODO: Complete
        save_spectogram(features, args.output_file, wave)

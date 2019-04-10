import torch
import torch.nn.functional as F
import librosa
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from scipy.signal import get_window
import scipy


_LOG_EPS = 1e-6


def stft(y, sr, n_fft=400, hop_t=0.010, win_t=0.025, window="hamming",
         preemphasis=0.97):
    """
    Short time Fourier Transform
    Args:
        y(np.ndarray): raw waveform of shape (T,)
        sr(int): sample rate
        hop_t(float): spacing (in second) between consecutive frames
        win_t(float): window size (in second)
        window(str): type of window applied for STFT
        preemphasis(float): pre-emphasize raw signal with y[t] = x[t] - r*x[t-1]
    Return:
        (np.ndarray): (n_fft / 2 + 1, N) matrix; N is number of frames
    """
    if preemphasis > 1e-12:
        y = y - preemphasis * np.concatenate([[0], y[:-1]], 0)
    hop_length = int(sr * hop_t)
    win_length = int(sr * win_t)
    return librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window)


def rstft(y, sr, n_fft=400, hop_t=0.010, win_t=0.025, window="hamming", 
        preemphasis=0.97, log=True, log_floor=-50):
    """
    Compute (log) magnitude spectrogram
    Args:
        y(np.ndarray): 
        sr(int):
        hop_t(float):
        win_t(float):
        window(str):
        preemphasis(float):
        log(bool):
    Return:
        (np.ndarray): (n_fft / 2 + 1, N) matrix; N is number of frames
    """
    spec = stft(y, sr, n_fft, hop_t, win_t, window, preemphasis)
    spec = np.abs(spec)
    if log:
        spec = np.log(spec)
        spec[spec < log_floor] = log_floor
    return spec


def to_melspec(y, sr, n_fft=400, hop_t=0.010, win_t=0.025, window="hamming", 
        preemphasis=0.97, n_mels=80, log=True, norm_mel=None, log_floor=-20):
    """
    Compute Mel-scale filter bank coefficients:
    Args:
        y(np.ndarray): 
        sr(int):
        hop_t(float):
        win_t(float):
        window(str):
        preemphasis(float):
        n_mels(int): number of filter banks, which are equally spaced in Mel-scale
        log(bool):
        norm_mel(None/1): normalize each filter bank to have area of 1 if set to 1;
            otherwise the peak value of eahc filter bank is 1
    Return:
        (np.ndarray): (n_mels, N) matrix; N is number of frames
    """
    spec = rstft(y, sr, n_fft, hop_t, win_t, window, preemphasis, log=False)
    hop_length = int(sr * hop_t)
    melspec = librosa.feature.melspectrogram(sr=sr, S=spec, n_fft=n_fft,
                                             hop_length=hop_length, n_mels=n_mels, norm=norm_mel)
    if log:
        melspec = np.log(melspec+1e-6)
        melspec[melspec < log_floor] = log_floor
    return melspec


def energy_vad(y, sr, hop_t=0.010, win_t=0.025, th_ratio=1.04/2):
    """
    Compute energy-based VAD
    """
    hop_length = int(sr * hop_t)
    win_length = int(sr * win_t)
    e = librosa.feature.rmse(y, frame_length=win_length, hop_length=hop_length)
    th = th_ratio * np.mean(e)
    vad = np.asarray(e > th, dtype=int)
    return vad


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=400, hop_length=160, win_length=400,
                 window='hamming'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = librosa.util.pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            torch.autograd.Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > librosa.util.tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio

            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length / 2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=400, hop_length=160, win_length=400,
                 n_mel_channels=80, sampling_rate=16000, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
            y = y.unsqueeze(0)
            y = torch.autograd.Variable(y, requires_grad=False)

        assert (torch.min(y.data) >= -1)
        assert (torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return torch.squeeze(mel_output, 0).detach().cpu().numpy().T

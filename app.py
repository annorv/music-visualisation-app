import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn plots
sns.set(style="whitegrid")

# Load the song
@st.cache_data
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

# Function to plot the waveform
def plot_waveform(y, sr):
    st.write("### Waveform of ASEDA ABBA - AWAKEN")
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title="Waveform")
    st.pyplot(fig)

# Function to plot the spectrogram
def plot_spectrogram(y, sr):
    st.write("### Spectrogram of ASEDA ABBA - AWAKEN")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Spectrogram")
    st.pyplot(fig)

# Function to plot the Mel spectrogram
def plot_mel_spectrogram(y, sr):
    st.write("### Mel Spectrogram of ASEDA ABBA - AWAKEN")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Mel Spectrogram")
    st.pyplot(fig)

# Function to plot beat tracking and tempo
def plot_beats_and_tempo(y, sr):
    st.write("### Beat Tracking and Tempo Estimation")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    st.write(f"Estimated Tempo: {tempo:.2f} BPM")

    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax)
    ax.vlines(librosa.frames_to_time(beats, sr=sr), -1, 1, color='r', linestyle='--', label='Beats')
    ax.set(title="Waveform with Beat Tracking")
    ax.legend()
    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title("Music Visualization - ASEDA ABBA")

    # Load the audio file from the data directory
    audio_path = 'data/ASEDA ABBA - AWAKEN final.mp3'
    y, sr = load_audio(audio_path)

    # Visualize different aspects of the audio
    plot_waveform(y, sr)
    plot_spectrogram(y, sr)
    plot_mel_spectrogram(y, sr)
    plot_beats_and_tempo(y, sr)

if __name__ == "__main__":
    main()

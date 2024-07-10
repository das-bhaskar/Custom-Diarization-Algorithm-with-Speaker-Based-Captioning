import os
import shutil
import torch
import torchaudio
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nemo.collections.asr as nemo_asr
import whisper
import streamlit as st
from moviepy.editor import VideoFileClip

# Step 1: Define functions for audio processing and embedding extraction
def load_audio(file_path, sample_rate=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    # Convert to mono if not already
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, sample_rate

def extract_embeddings(model, audio):
    audio = audio.to(model.device)
    model.eval()
    with torch.no_grad():
        # Remove the channel dimension if it's mono
        if audio.shape[0] == 1:
            audio = audio.squeeze(0)
        # Expand dimensions to add batch dimension
        audio = audio.unsqueeze(0)
        input_signal_length = torch.tensor([audio.shape[1]], device=audio.device)

        _, embeddings = model.forward(input_signal=audio, input_signal_length=input_signal_length)
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings.squeeze()  # Ensure embeddings are 1D

        # Custom normalization to handle zero norm
        norm = np.linalg.norm(embeddings)
        if norm == 0:
            print("Warning: Embedding norm is zero.")
            return np.zeros_like(embeddings)
        embeddings = embeddings / norm
    return embeddings

# Step 2: Load the pretrained model
verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")

# Step 3: Define functions for diarization, clustering, and transcription
def load_known_embeddings(csv_path):
    df = pd.read_csv(csv_path, header=None)
    labels = df.iloc[:, 0].values
    embeddings = df.iloc[:, 1:].values
    return labels, embeddings

def extract_embeddings_from_windows(audio_file, window_size=0.5, overlap=0.25, sample_rate=16000):
    audio, sr = load_audio(audio_file, sample_rate)
    window_samples = int(window_size * sample_rate)
    step_samples = int((window_size - overlap) * sample_rate)
    embeddings = []

    for start in range(0, audio.shape[1] - window_samples + 1, step_samples):
        window = audio[:, start:start + window_samples]
        embedding = extract_embeddings(verification_model, window)
        embeddings.append(embedding)

    return np.array(embeddings)

def cluster_and_label(embeddings, known_labels, known_embeddings, num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    cluster_labels = kmeans.labels_

    # Assign speaker labels based on highest cosine similarity with known embeddings
    speaker_labels = []
    for cluster_center in kmeans.cluster_centers_:
        similarities = cosine_similarity([cluster_center], known_embeddings)
        best_match_index = np.argmax(similarities)
        speaker_labels.append(known_labels[best_match_index])

    return cluster_labels, speaker_labels

def diarize_audio(audio_file, csv_path, output_dir, window_size=0.5, overlap=0.25, sample_rate=16000):
    known_labels, known_embeddings = load_known_embeddings(csv_path)
    audio, sr = load_audio(audio_file, sample_rate)
    embeddings = extract_embeddings_from_windows(audio_file, window_size, overlap, sample_rate)
    cluster_labels, speaker_labels = cluster_and_label(embeddings, known_labels, known_embeddings)

    # Initialize variables for segment merging
    segments = []
    current_start = 0.0
    current_end = 0.0
    current_speaker = None

    # Map clusters to speaker labels and merge contiguous segments
    for i, cluster_label in enumerate(cluster_labels):
        start_time = i * int((window_size - overlap) * sample_rate) / sample_rate
        end_time = start_time + window_size

        if current_speaker is None:
            current_speaker = speaker_labels[cluster_label]
            current_start = start_time
            current_end = end_time
        elif speaker_labels[cluster_label] == current_speaker:
            current_end = end_time
        else:
            segments.append((current_start, current_end, current_speaker))
            current_speaker = speaker_labels[cluster_label]
            current_start = start_time
            current_end = end_time

    # Append the last segment
    if current_speaker is not None:
        segments.append((current_start, current_end, current_speaker))

    # Save segments as WAV files and transcribe them
    results = []
    model = whisper.load_model("medium")
    for idx, (start, end, speaker) in enumerate(segments):
        segment_file_path = os.path.join(output_dir, f"segment_{idx}.wav")
        torchaudio.save(segment_file_path, audio[:, int(start * sample_rate):int(end * sample_rate)], sample_rate)
        
        # Transcribe using Whisper
        result = model.transcribe(segment_file_path)
        transcription = result["text"]

        # Store results
        results.append((start, end, speaker, transcription))

    return results

# Function to handle MP4 file conversion
def convert_mp4_to_wav(mp4_file_path, output_dir):
    video = VideoFileClip(mp4_file_path)
    audio_file_path = os.path.join(output_dir, "audio.wav")
    video.audio.write_audiofile(audio_file_path, codec="pcm_s16le")
    return audio_file_path

# Streamlit app
st.title("Audio Diarization and Transcription")

if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = []
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

uploaded_file = st.file_uploader("Upload a WAV or MP4 file", type=["wav", "mp4"])
if uploaded_file is not None:
    audio_file = uploaded_file.name
    file_extension = os.path.splitext(audio_file)[1].lower()

    # Only process the audio if a new file is uploaded
    if st.session_state.uploaded_file_name != audio_file:
        csv_path = "embeddings.csv"
        output_dir = "outputsegments"

        # Clear the output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        with open(os.path.join(output_dir, audio_file), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Convert MP4 to WAV if necessary
        if file_extension == ".mp4":
            audio_file_path = convert_mp4_to_wav(os.path.join(output_dir, audio_file), output_dir)
        else:
            audio_file_path = os.path.join(output_dir, audio_file)

        segments = diarize_audio(audio_file_path, csv_path, output_dir)

        # Store transcriptions in session state
        st.session_state.transcriptions = [(start, end, speaker, transcription) for start, end, speaker, transcription in segments]
        st.session_state.uploaded_file_name = audio_file

# Display the transcriptions
if st.session_state.transcriptions:
    st.header("Transcriptions")
    for start, end, speaker, transcription in st.session_state.transcriptions:
        st.write(f"\n[{start:.2f}s - {end:.2f}s] \n **{speaker}**: {transcription}\n")

# Add a search bar
search_query = st.text_input("Search for a word:")
if st.button("Search"):
    st.header("Search Results")
    for start, end, speaker, transcription in st.session_state.transcriptions:
        if search_query.lower() in transcription.lower():
            st.write(f"\n[{start:.2f}s - {end:.2f}s] \n **{speaker}**: {transcription}\n")

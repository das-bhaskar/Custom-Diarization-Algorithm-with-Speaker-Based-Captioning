import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import nemo.collections.asr as nemo_asr

# Step 2: Define functions to process audio files and extract embeddings
def load_audio(file_path, sample_rate=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    # Convert to mono if not already
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, sample_rate

def extract_embeddings(model, audio_file):
    audio, sr = load_audio(audio_file)
    audio = audio.to(model.device)
    model.eval()
    with torch.no_grad():
        # Remove the channel dimension if it's mono
        if audio.shape[0] == 1:
            audio = audio.squeeze(0)
        # Expand dimensions to add batch dimension
        audio = audio.unsqueeze(0)
        input_signal_length = torch.tensor([audio.shape[1]], device=audio.device)
        print(f"Audio shape: {audio.shape}, input_signal_length: {input_signal_length}")

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

# Step 3: Load the pretrained model
verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")

# Step 7: Save embeddings to a CSV file
def save_embeddings_to_csv(file_paths, csv_path="embeddings.csv"):
    embeddings_list = []
    for file_path in file_paths:
        embeddings = extract_embeddings(verification_model, file_path)
        label = os.path.splitext(os.path.basename(file_path))[0]  # Remove the .wav extension
        embeddings_list.append([label] + embeddings.tolist())

    df = pd.DataFrame(embeddings_list)
    if os.path.exists(csv_path):
        df.to_csv(csv_path, index=False, header=False, mode='a')
    else:
        df.to_csv(csv_path, index=False, header=False)

file_paths = [
    "path/to/wav/file/of/persons/voice/sample"
]
save_embeddings_to_csv(file_paths)

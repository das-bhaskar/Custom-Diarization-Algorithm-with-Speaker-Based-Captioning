<div align="center">
  <img src="https://placehold.it/200x50/2ecc71/ffffff?text=Audio+Diarization" alt="Audio Diarization Logo">
</div>

---

# <span style="color:#2ecc71;">Audio Diarization and Transcription</span>

## <span style="color:#3498db;">Introduction</span>
This folder contains scripts and a Streamlit web application designed for audio diarization and transcription tasks. Audio diarization involves segmenting an audio stream into homogeneous segments, each associated with a specific speaker, while transcription converts these segments into text.

## <span style="color:#3498db;">Overview</span>

This project focuses on audio diarization and transcription, primarily aimed at processing audio extracted from video files. It utilizes several Python libraries and tools such as `moviepy` for video processing, `torchaudio` for audio manipulation, `NeMo` for speaker embedding extraction, `KMeans` for clustering, and `whisper` for transcription. The project also includes a Streamlit web application for a user-friendly interface to upload files, perform diarization, transcription, and view results interactively.

### <span style="color:#3498db;">Files</span>

- `audioextract.py`: Script to extract audio from a video file.
- `store_in_csv.py`: Script to extract speaker embeddings from audio files and save them to a CSV file.
- `streamlit_diarized_caption.py`: Streamlit application for uploading files, performing diarization and transcription, and displaying the results.
- `dhairyasaahil.mp4`: Example video file.
- `embeddings.csv`: CSV file containing precomputed speaker embeddings.
- `requirements.txt`: List of required Python packages.

## <span style="color:#3498db;">How to Run</span>

### <span style="color:#e74c3c;">Setup</span>

1. **Install Required Packages:**
   - Ensure Python 3.8 or higher is installed.
   - Install necessary packages using pip:

     ```bash
     pip install -r requirements.txt
     ```

### <span style="color:#e74c3c;">Usage</span>

1. **Extract Audio from Video (`audioextract.py`):**
   - Modify `video_path` variable in `audioextract.py` to specify the path of your video file.
   - Run the script:

     ```bash
     python audioextract.py
     ```

   - This script extracts audio from a video file and saves it as a WAV file (`audio.wav`).

2. **Extract Speaker Embeddings (`store_in_csv.py`):**
   - Ensure your audio files are in WAV format.
   - Modify `file_paths` variable in `store_in_csv.py` to include paths to your audio files.
   - Run the script:

     ```bash
     python store_in_csv.py
     ```

   - This script extracts speaker embeddings using a pretrained model and saves them to `embeddings.csv`.

3. **Run the Streamlit Application (`streamlit_diarized_caption.py`):**
   - Execute the Streamlit web application:

     ```bash
     streamlit run streamlit_diarized_caption.py
     ```

   - Upload a WAV or MP4 file using the provided file uploader.
   - The application performs audio diarization and transcription, displaying results including speaker labels and timestamps.
   - Use the search bar to find specific words in the transcriptions.

## <span style="color:#3498db;">What to Change</span>

- **File Paths:**
  - Modify `video_path` in `audioextract.py` to point to your video file.
  - Adjust `file_paths` in `store_in_csv.py` to include paths to your audio files.
- **CSV Path:**
  - Ensure `embeddings.csv` is in the same directory as the scripts or modify the path accordingly in `store_in_csv.py`.

## <span style="color:#3498db;">Logic Behind Each Script</span>

### <span style="color:#e74c3c;">`audioextract.py`</span>

#### <span style="color:#e67e22;">Logic:</span>

1. **Purpose:** Extract audio from a video file.

2. **Steps:**
   - **Import Libraries:** Import necessary libraries, including `moviepy.editor` for video processing.
   - **Define Video Path:** Specify the path to the input video file (`video_path`).
   - **Load Video:** Use `VideoFileClip` from `moviepy.editor` to load the video specified by `video_path`.
   - **Extract Audio:** Access the audio component of the video clip using `video_clip.audio`.
   - **Save Audio:** Write the audio to a WAV file using `audio_clip.write_audiofile("Ekaansh.wav")`.

3. **Detailed Breakdown:**
   - The script starts by importing the necessary library `moviepy.editor`.
   - It defines `video_path` to point to the location of the input video file.
   - The video is loaded using `VideoFileClip(video_path)`, which creates a `VideoFileClip` object representing the video file.
   - `video_clip.audio` extracts the audio part of the video clip.
   - `audio_clip.write_audiofile("Ekaansh.wav")` saves the extracted audio as a WAV file named "Ekaansh.wav".

### <span style="color:#e74c3c;">`store_in_csv.py`</span>

#### <span style="color:#e67e22;">Logic:</span>

1. **Purpose:** Extract speaker embeddings from audio files and store them in a CSV file.

2. **Steps:**
   - **Import Libraries:** Import required libraries, including `torch`, `torchaudio`, `numpy`, `pandas`, and `nemo.collections.asr`.
   - **Define Audio Loading Function (`load_audio`):** `load_audio` loads an audio file, performs resampling if necessary, and converts stereo to mono.
   - **Define Embedding Extraction Function (`extract_embeddings`):** `extract_embeddings` takes an audio waveform, extracts embeddings using a pretrained speaker recognition model (`verification_model` from `nemo_asr`), normalizes them, and returns them as a numpy array.
   - **Load Pretrained Model:** Load the speaker recognition model (`verification_model`) using `nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained`.
   - **Define CSV Saving Function (`save_embeddings_to_csv`):** `save_embeddings_to_csv` processes multiple audio files, extracts embeddings using `extract_embeddings`, and saves them along with file labels to a CSV file (`embeddings.csv`).

3. **Detailed Breakdown:**
   - **Audio Loading (`load_audio`):**
     - Uses `torchaudio.load` to load audio data and its sample rate.
     - Performs resampling if the sample rate is different from the desired rate (`sample_rate`).
     - Converts stereo audio to mono by averaging across channels if necessary.
   - **Embedding Extraction (`extract_embeddings`):**
     - Converts the audio waveform to a tensor and moves it to the device where `verification_model` is loaded.
     - Uses the model to extract embeddings (`model.forward`), handling mono audio by adjusting dimensions.
     - Normalizes embeddings to ensure they are scaled appropriately.
   - **CSV Saving (`save_embeddings_to_csv`):**
     - Reads a list of file paths (`file_paths`).
     - For each file, calls `extract_embeddings` to obtain embeddings and appends them to `embeddings_list`.
     - Creates or appends to a CSV file (`embeddings.csv`) with columns for file labels and embeddings.

### <span style="color:#e74c3c;">`streamlit_diarized_caption.py`</span>

#### <span style="color:#e67e22;">Logic:</span>

1. **Purpose:** Implement a Streamlit web application for audio diarization and transcription.

2. **Steps:**
   - **Import Libraries:** Import necessary libraries, including `os`, `shutil`, `torch`, `torchaudio`, `numpy`, `pandas`, `sklearn`, `nemo.collections.asr`, `whisper`, and `streamlit`.
   - **Define Audio Processing Functions (`load_audio`, `extract_embeddings`, `convert_mp4_to_wav`):**
     - `load_audio`: Loads audio from a file, resamples if needed, and converts stereo to mono.
     - `extract_embeddings`: Uses `verification_model` to extract speaker embeddings from audio segments.
     - `convert_mp4_to_wav`: Converts MP4 video files to WAV audio files using `moviepy`.
   - **Load Pretrained Model (`verification_model`):** Loads the speaker recognition model using `nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained`.
   - **Define Diarization and Transcription Functions (`diarize_audio`):**
     - `diarize_audio`: Utilizes clustering (via `KMeans`) to segment audio based on speaker embeddings, identifies speaker labels, saves segments as WAV files, and transcribes them using `whisper`.
   - **Implement Streamlit Application (`streamlit`):**
     - Initializes a Streamlit web application (`st.title`, `st.file_uploader`, `st.session_state`).
     - Handles file uploading, processing (including audio conversion if necessary), diarization, and transcription.
     - Displays results including timestamps, speaker labels, and transcriptions.
     - Implements a search feature to find specific words in transcriptions.

3. **Detailed Breakdown:**
   - **Audio Processing (`load_audio`, `extract_embeddings`, `convert_mp4_to_wav`):**
     - `load_audio`: Loads audio data, adjusts sample rate if necessary, and ensures mono audio for consistency.
     - `extract_embeddings`: Uses a pretrained model (`verification_model`) to extract embeddings from audio segments, ensuring normalization for consistent scaling.
     - `convert_mp4_to_wav`: Converts MP4 video files to WAV audio files to facilitate audio processing.
   - **Diarization and Transcription (`diarize_audio`):**
     - Segments audio using `KMeans` clustering based on speaker embeddings to identify speaker changes over time.
     - Merges contiguous segments belonging to the same speaker to enhance readability.
     - Transcribes audio segments using `whisper`, handling transcription errors or failures gracefully.
   - **Streamlit Application:**
     - Provides an intuitive user interface for uploading audio or video files (`st.file_uploader`).
     - Processes uploaded files, performing necessary conversions (`convert_mp4_to_wav`) and invoking diarization/transcription (`diarize_audio`).
     - Displays results interactively, allowing users to explore timestamps, speaker labels, and transcriptions.
     - Implements search functionality (`st.text_input`, `st.button`) to facilitate keyword search within transcriptions.

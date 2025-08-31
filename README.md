## 📖 TTS Dataset Pipeline

A simple three-step pipeline for building paired (audio, text) datasets for TTS / ASR tasks.

### Core workflow:

1. VAD Splitting – use Silero VAD
 to segment long audio into short utterances.

2. Whisper Transcription – use OpenAI Whisper
 to generate transcriptions.

3. Upload to Hugging Face Hub – package the dataset into standard 🤗 Datasets
 format.

The final dataset structure:

```bash
my_dataset/
  ├── data/
  │    ├── chunk_0001.wav
  │    ├── chunk_0002.wav
  │    └── ...
  └── metadata.csv

```
```metadata.csv``` example:

```bash
file,text
data/chunk_0001.wav,This is the first sentence.
data/chunk_0002.wav,This is the second one.
...

```

### 🚀 Quickstart
1. Install dependencies (using uv)


```bash
uv pip install -r pyproject.toml
or
uv sync
```
Main dependencies:
- torch, torchaudio

- openai-whisper

- datasets, huggingface_hub

2. VAD-based audio splitting
```bash
python vad_split.py --input demo.wma --output my_dataset
```
This will create segmented audio chunks under my_dataset/data/

3. Whisper transcription
```bash
python transcribe.py --dataset my_dataset --model_size medium --lang zh

```
This generates metadata.csv inside my_dataset/, containing paired (file, text) entries.
4. Upload to Hugging Face Hub
```bash
python upload_hf.py --dataset my_dataset --repo your-username/your-dataset-name

```
You will be prompted for your Hugging Face token.

## 📂 Project Structure

```bash
project/
│── pyproject.toml     # uv dependency management
│── vad_split.py       # Step 1: VAD splitting
│── transcribe.py      # Step 2: Whisper transcription
│── upload_hf.py       # Step 3: Upload to Hugging Face
│── my_dataset/        # Output dataset
│    ├── data/
│    └── metadata.csv


```
## 📝 License

MIT License
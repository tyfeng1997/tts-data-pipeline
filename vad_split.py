import torch
import torchaudio
from pathlib import Path

def merge_short_segments(timestamps, min_duration=24000):
    """Merge segments shorter than min_duration into the previous one"""
    merged = []
    for seg in timestamps:
        if merged and seg['end'] - seg['start'] < min_duration:
            merged[-1]['end'] = seg['end']
        else:
            merged.append(seg)
    return merged

def vad_split(input_path: str, output_dir: str):
    """Run VAD on a single file or all files in a directory"""
    model, utils = torch.hub.load("snakers4/silero-vad", model="silero_vad")
    (get_speech_timestamps, _, read_audio, _, _) = utils

    input_path = Path(input_path)
    output_dir = Path(output_dir) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support both single file and directory
    if input_path.is_file():
        audio_files = [input_path]
    else:
        audio_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.wma")) + list(input_path.glob("*.mp3"))

    if not audio_files:
        print(f"No audio files found in {input_path}")
        return

    global_chunk_idx = 1
    for audio_file in audio_files:
        print(f"Processing {audio_file} ...")
        wav = read_audio(str(audio_file), sampling_rate=16000)
        timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
        merged = merge_short_segments(timestamps)

        for seg in merged:
            start, end = seg["start"], seg["end"]
            chunk = wav[start:end].unsqueeze(0)
            file_path = output_dir / f"chunk_{global_chunk_idx:04d}.wav"
            torchaudio.save(str(file_path), chunk, 16000)
            global_chunk_idx += 1

    print(f"✅ Saved {global_chunk_idx-1} chunks to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input audio file or directory")
    parser.add_argument("--output", default="my_dataset", help="Output dataset directory")
    args = parser.parse_args()
    vad_split(args.input, args.output)

import torch
import torchaudio

def merge_short_segments(timestamps, min_duration=24000):
    
    merged = []
    for seg in timestamps:
        if merged and seg['end'] - seg['start'] < min_duration:
            merged[-1]['end'] = seg['end']
        else:
            merged.append(seg)
    return merged

def vad_split(input_file: str, output_dir: str):
    model, utils = torch.hub.load("snakers4/silero-vad", model="silero_vad")
    (get_speech_timestamps, _, read_audio, _, _) = utils

    wav = read_audio(input_file, sampling_rate=16000)
    timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
    merged = merge_short_segments(timestamps)

    output_dir = Path(output_dir) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(merged, 1):
        start, end = seg["start"], seg["end"]
        chunk = wav[start:end].unsqueeze(0)
        file_path = output_dir / f"chunk_{i:04d}.wav"
        torchaudio.save(str(file_path), chunk, 16000)

    print(f"Saved {len(merged)} chunks to {output_dir}")

if __name__ == "__main__":
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--output", default="my_dataset", help="Output dataset directory")
    args = parser.parse_args()
    vad_split(args.input, args.output)

import whisper
import csv
from pathlib import Path

def transcribe_dataset(dataset_dir: str, model_size="medium", lang="zh"):
    dataset_dir = Path(dataset_dir)
    data_dir = dataset_dir / "data"
    model = whisper.load_model(model_size, device="cuda")

    metadata_path = dataset_dir / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "text"])

        for wav_file in sorted(data_dir.glob("*.wav")):
            result = model.transcribe(str(wav_file), language=lang)
            writer.writerow([str(wav_file), result["text"]])
            print(f"{wav_file.name}: {result['text']}")

    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="my_dataset", help="Dataset directory")
    parser.add_argument("--model_size", default="medium", help="Whisper model size")
    parser.add_argument("--lang", default="zh", help="Language code")
    args = parser.parse_args()
    transcribe_dataset(args.dataset, args.model_size, args.lang)

from datasets import load_dataset, Audio
from huggingface_hub import login

def upload_to_hf(dataset_dir: str, repo_id: str, sampling_rate=16000):
    # 登录
    login()  # 首次运行会提示输入 token

    # 加载 CSV + 音频
    ds = load_dataset("csv", data_files=f"{dataset_dir}/metadata.csv")
    ds = ds.cast_column("file", Audio(sampling_rate=sampling_rate))
    ds = ds.rename_column("file", "audio")

    # 推送
    ds.push_to_hub(repo_id)
    print(f"✅ Uploaded dataset to https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="my_dataset", help="Dataset directory")
    parser.add_argument("--repo", required=True, help="Hugging Face repo id (username/dataset_name)")
    args = parser.parse_args()
    upload_to_hf(args.dataset, args.repo)

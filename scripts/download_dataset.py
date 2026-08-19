from pathlib import Path
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESTINATION = PROJECT_ROOT / "data" / "raw" / "MSMARCO-XI"

REPO_ID = "ai4bharat/MSMARCO-XI"

FILES = [
    "train/hintrain.parquet",
    "train/martrain.parquet",
]


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MSMARCO-XI DATASET DOWNLOAD")
    print("=" * 70)
    print(f"Destination: {DESTINATION}")
    print()

    for file_path in FILES:
        print("-" * 70)
        print(f"Downloading: {file_path}")
        print("-" * 70)

        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            repo_type="dataset",
            local_dir=str(DESTINATION),
        )

        print(f"Downloaded: {downloaded_path}")
        print()

    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    for file_path in FILES:
        local_path = DESTINATION / file_path
        if local_path.exists():
            size_gb = local_path.stat().st_size / (1024 ** 3)
            print(f"{file_path}: {size_gb:.2f} GB")
        else:
            print(f"MISSING: {file_path}")


if __name__ == "__main__":
    main()
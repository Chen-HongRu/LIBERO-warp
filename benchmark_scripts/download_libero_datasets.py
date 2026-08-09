import argparse
import os
import time

from libero.libero import get_libero_path
from libero.libero.utils import download_utils


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download LIBERO demonstration datasets."
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Destination directory (defaults to the configured LIBERO datasets path).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        choices=[
            "all",
            "libero_goal",
            "libero_spatial",
            "libero_object",
            "libero_100",
        ],
        default="all",
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Use Hugging Face instead of the original download links.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.download_dir is None:
        args.download_dir = get_libero_path("datasets")

    # Ask users to specify the download directory of datasets
    os.makedirs(args.download_dir, exist_ok=True)
    print(f"Datasets downloaded to {args.download_dir}")
    print(f"Downloading {args.datasets} datasets")

    if args.use_huggingface:
        print("Using Hugging Face as the download source")
    else:
        print("Using original download links (note: these may expire soon)")
        input_str = input(
            "Download from original links may lead to failures. Continue? (y/n): "
        )
        if input_str.lower() != "y":
            print("Switching to Hugging Face as the download source...")
            args.use_huggingface = True

    # If not, download
    download_utils.libero_dataset_download(
        download_dir=args.download_dir,
        datasets=args.datasets,
        use_huggingface=args.use_huggingface,
    )

    # Allow filesystems with delayed writes to settle before the integrity check.
    time.sleep(1)
    print("\n\n\n")

    # Check if datasets exist first
    download_utils.check_libero_dataset(download_dir=args.download_dir)


if __name__ == "__main__":
    main()

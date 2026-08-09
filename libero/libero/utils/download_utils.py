"""
Download functionalities adapted from Mandlekar et. al.: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/file_utils.py
"""

import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from termcolor import colored
from tqdm import tqdm

from libero.libero import get_libero_path

try:
    from huggingface_hub import snapshot_download

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


class DownloadProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    From https://gist.github.com/dehowell/884204.
    Args:
        url (str): url string
    Returns:
        is_alive (bool): True if url is reachable, False otherwise
    """
    request = urllib.request.Request(url)
    # request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def download_url(url, download_dir, check_overwrite=True, is_zipfile=True):
    """
    First checks that @url is reachable, then downloads the file
    at that url into the directory specified by @download_dir.
    Prints a progress bar during the download using tqdm.
    Modified from https://github.com/tqdm/tqdm#hooks-and-callbacks, and
    https://stackoverflow.com/a/53877507.
    Args:
        url (str): url string
        download_dir (str): path to directory where file should be downloaded
        check_overwrite (bool): Whether to ensure that the destination file does
            not already exist.
    """

    # The short delay prevents some servers from rejecting the following request.
    assert url_is_alive(url), f"@download_url got unreachable url: {url}"
    time.sleep(0.5)

    # infer filename from url link
    fname = url.split("/")[-1]
    file_to_write = os.path.join(download_dir, fname)

    # If we're checking overwrite and the path already exists,
    # we ask the user to verify that they want to overwrite the file
    user_response = None
    if check_overwrite and os.path.exists(file_to_write):
        user_response = input(
            f"Warning: file {file_to_write} already exists. Overwrite? y/n\n"
        )

    if user_response is None or user_response.lower() in {"yes", "y"}:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=fname
        ) as t:
            urllib.request.urlretrieve(
                url, filename=file_to_write, reporthook=t.update_to
            )
    if is_zipfile:
        with zipfile.ZipFile(file_to_write, "r") as archive:
            archive.extractall(path=download_dir)
        if os.path.isfile(file_to_write):
            os.remove(file_to_write)


DATASET_LINKS = {
    "libero_object": "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip",
    "libero_goal": "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip",
    "libero_spatial": "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip",
    "libero_100": "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip",
}

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"


def download_from_huggingface(dataset_name, download_dir, check_overwrite=True):
    """
    Download a specific LIBERO dataset from Hugging Face.

    Args:
        dataset_name (str): Name of the dataset to download (e.g., 'libero_spatial')
        download_dir (str): Directory where the dataset should be downloaded
        check_overwrite (bool): If True, will check if dataset already exists
    """
    if not HUGGINGFACE_AVAILABLE:
        raise ImportError(
            "Hugging Face Hub is not available. Install the optional dependency "
            "with 'uv sync --extra huggingface'."
        )

    # Create the destination folder
    os.makedirs(download_dir, exist_ok=True)

    # Check if dataset already exists
    dataset_dir = os.path.join(download_dir, dataset_name)
    if check_overwrite and os.path.exists(dataset_dir):
        user_response = input(
            "Warning: dataset "
            f"{dataset_name} already exists at {dataset_dir}. Overwrite? y/n\n"
        )
        if user_response.lower() not in {"yes", "y"}:
            print(f"Skipping download of {dataset_name}")
            return

        # Remove existing directory
        print(f"Removing existing folder: {dataset_dir}")
        shutil.rmtree(dataset_dir)

    # Download the dataset
    print(f"Downloading {dataset_name} from Hugging Face...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=f"{dataset_name}/*",
    )

    # Verify downloaded files
    file_count = sum(
        [
            len(files)
            for _, _, files in os.walk(os.path.join(download_dir, dataset_name))
        ]
    )
    print(f"Downloaded {file_count} files for {dataset_name}")


def libero_dataset_download(
    datasets="all", download_dir=None, check_overwrite=True, use_huggingface=False
):
    """Download libero datasets

    Args:
        datasets (str, optional): Dataset to save, or ``"all"`` for every suite.
        download_dir (str, optional): Destination. Defaults to the configured
            LIBERO datasets path.
        check_overwrite (bool, optional): Whether to check before overwriting.
        use_huggingface (bool, optional): Use Hugging Face rather than the
            original download links.
    """
    if download_dir is None:
        download_dir = get_libero_path("datasets")
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    assert datasets in [
        "all",
        "libero_object",
        "libero_goal",
        "libero_spatial",
        "libero_100",
    ]

    datasets_to_download = (
        [
            "libero_object",
            "libero_goal",
            "libero_spatial",
            "libero_100",
        ]
        if datasets == "all"
        else [datasets]
    )

    for dataset_name in datasets_to_download:
        print(f"Downloading {dataset_name}")

        if use_huggingface:
            download_from_huggingface(
                dataset_name=dataset_name,
                download_dir=download_dir,
                check_overwrite=check_overwrite,
            )
        else:
            print("Using original download links (these may expire soon)")
            download_url(
                DATASET_LINKS[dataset_name],
                download_dir=download_dir,
                check_overwrite=check_overwrite,
            )


def check_libero_dataset(download_dir=None):
    """Check the integrity of the downloaded datasets.

    Args:
        download_dir (str, optional): Dataset directory. Defaults to the
            configured LIBERO datasets path.

    Returns:
        bool: True if the datasets are successfully downloaded, False otherwise.
    """
    if download_dir is None:
        download_dir = get_libero_path("datasets")
    check_result = True
    for dataset_name in [
        "libero_object",
        "libero_goal",
        "libero_spatial",
        "libero_10",
        "libero_90",
    ]:
        info_str = ""
        dataset_status = False
        dataset_dir = os.path.join(download_dir, dataset_name)
        if os.path.exists(dataset_dir):
            count = 0
            for path in Path(dataset_dir).glob("*.hdf5"):
                count += 1
            if (count == 10 and dataset_name != "libero_90") or (
                count == 90 and dataset_name == "libero_90"
            ):
                dataset_status = True
                info_str = colored(
                    f"[X] Dataset {dataset_name} is complete", "green", attrs=["bold"]
                )
            else:
                colored(
                    f"[?] Dataset {dataset_name} is not downloaded completely",
                    "yellow",
                    attrs=["bold"],
                )
        else:
            info_str = colored(
                f"[ ] Dataset {dataset_name} not found!!!", "red", attrs=["bold"]
            )

        print(info_str)
        check_result = check_result and dataset_status
    return check_result

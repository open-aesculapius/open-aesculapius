import os
import requests
from tqdm import tqdm
from pathlib import Path
from typing import List


WEIGHTS_URL = "OpenAesculapius/OpenAesculapius"

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "examples" or PROJECT_ROOT.name == 'test':
    WEIGHTS_ROOT = PROJECT_ROOT.parent / "weights"
else:
    WEIGHTS_ROOT = PROJECT_ROOT / "weights"


def get_file_list_from_hf(repo_id: str = WEIGHTS_URL,
                          folder: str = "") -> List[str]:

    folder = os.path.basename(folder)

    url = f"https://huggingface.co/api/models/{repo_id}/tree/main/{folder}"
    response = requests.get(url)
    response.raise_for_status()

    files = []
    for item in response.json():
        if item["type"] == "file":
            files.append(item["path"])
        elif item["type"] == "directory":
            files.extend(get_file_list_from_hf(repo_id, item["path"]))

    return files


def download_file_from_hf(
    filename: str, local_dir: Path = WEIGHTS_ROOT
) -> str:
    filename = os.path.basename(filename)

    os.makedirs(local_dir, exist_ok=True)

    local_dir = local_dir / filename

    if os.path.exists(local_dir):
        return str(local_dir)

    url = f"https://huggingface.co/{WEIGHTS_URL}/resolve/main/{filename}"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(local_dir, "wb") as file, tqdm(
        desc=f"Скачивание {filename}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

    return str(local_dir)


def download_folder_from_hf(
    folder: str = "",
    repo_id: str = WEIGHTS_URL,
    local_dir: str = WEIGHTS_ROOT,
    force_redownload: bool = False,
) -> List[str]:

    files = get_file_list_from_hf(repo_id, folder)

    downloaded_files = []
    existing_files = 0

    for file in files:
        local_path = Path(os.path.join(local_dir, file))

        if not force_redownload and local_path.exists():
            existing_files += 1
            downloaded_files.append(str(local_path))
            continue
        try:
            local_path = download_file_from_hf(file, repo_id, local_dir)
            downloaded_files.append(str(local_path))
        except Exception as e:
            print(f"[ERROR] Ошибка при скачивании {file}: {str(e)}")

    return downloaded_files

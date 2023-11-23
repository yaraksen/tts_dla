from torch.hub import download_url_to_file

def download_asset(url: str) -> str:
    # original function is not available in torchaudio=0.11
    path = f'data/{url.split("/")[1]}'
    url = f"https://download.pytorch.org/torchaudio/{url}"
    download_url_to_file(url, path)
    return path
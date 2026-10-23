"""Download the Chatbot Arena human-preference data into data/raw/.

The data is CC BY-NC 4.0 and must never be committed (data/ is gitignored and
`make leak-check` guards CI).
"""

from huggingface_hub import snapshot_download

REPO_ID = "lmsys/lmsys-arena-human-preference-55k"


def main() -> None:
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir="data/raw")


if __name__ == "__main__":
    main()

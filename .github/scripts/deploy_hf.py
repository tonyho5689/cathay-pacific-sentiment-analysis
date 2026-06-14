"""Deploy this repo to a Hugging Face Space from CI.

Runs on a GitHub Actions runner (which has internet access to huggingface.co,
unlike some sandboxed environments). It creates the Streamlit Space if it does
not exist yet, then uploads the app files. Authentication uses the HF_TOKEN
repository secret — never hard-code a token here.
"""

import os
import sys

from huggingface_hub import HfApi

# Space to deploy to. Override with the SPACE_REPO_ID env var if needed.
REPO_ID = os.environ.get(
    "SPACE_REPO_ID", "tonyho5689/cathay-pacific-sentiment-analysis"
)

# Files/folders that should NOT be uploaded to the Space.
IGNORE_PATTERNS = [
    ".git/**",
    ".github/**",
    ".devcontainer/**",
    "notebooks/**",
    "*.ipynb",
    "DEPLOY_HF.md",
]


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN secret is not set. Add it in the GitHub repo under "
            "Settings -> Secrets and variables -> Actions -> New repository secret "
            "(name: HF_TOKEN, value: a Hugging Face *write* token).",
            file=sys.stderr,
        )
        return 1

    api = HfApi(token=token)

    # HF no longer supports a native Streamlit SDK; we run Streamlit via Docker.
    print(f"Ensuring Space {REPO_ID} exists (Docker SDK)...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    print("Uploading app files to the Space...")
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=".",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Deploy from GitHub Actions",
    )

    print(f"Done. Space: https://huggingface.co/spaces/{REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Deploying to Hugging Face Spaces

This app runs on **Hugging Face Spaces** using the **Docker SDK**. (Hugging Face
no longer offers a native Streamlit SDK, so the Streamlit app runs inside the
`Dockerfile` in this repo, listening on port 7860.)

Spaces is a good home for this project because:

- Your fine-tuned model already lives on the HF Hub.
- Free CPU Spaces have **16 GB RAM / 2 vCPU** — plenty for Whisper-small + the
  sentiment model.
- A free Space only **pauses after 48 hours of zero traffic** (and wakes on the
  next visit) — far gentler than Streamlit Community Cloud's aggressive sleep.

Everything needed is already in this repo:

| File | Purpose |
|------|---------|
| `README.md` (YAML frontmatter) | `sdk: docker`, `app_port: 7860` — tells Spaces to build the Dockerfile |
| `Dockerfile` | Installs deps + runs `streamlit run app.py` on port 7860 |
| `app.py` | The Streamlit application |
| `requirements.txt` | Python dependencies (installed inside the image) |
| `.streamlit/config.toml` | Theme + 50 MB upload limit |

> Note: system packages (`ffmpeg`, `libsndfile1`) are installed by the
> `Dockerfile`. `packages.txt` is only used by non-Docker SDKs and is ignored here.

---

## Easiest: automatic deploy from GitHub Actions (recommended)

This repo includes `.github/workflows/deploy-hf.yml`, which creates the Space and
uploads the app on every push to `main`.

1. Create a **write** token: <https://huggingface.co/settings/tokens>.
2. Add it as a GitHub repo secret named **`HF_TOKEN`**
   (**Settings → Secrets and variables → Actions → New repository secret**).
3. Push to `main` (or run **Actions → Deploy to Hugging Face Space → Run workflow**).

The workflow builds the Space at
`https://huggingface.co/spaces/<user>/cathay-pacific-sentiment-analysis`.

---

## Manual: create the Space and push with git

1. **Create the Space** at <https://huggingface.co/new-space>:
   - **SDK:** **Docker** → template **Blank**.
   - **Hardware:** **CPU basic** (free) · **Visibility:** Public.
2. **Get a write token:** <https://huggingface.co/settings/tokens> → role **Write**.
3. **Push this repo to the Space:**

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/cathay-pacific-sentiment-analysis
   git push space main --force
   ```
   - **Username:** your Hugging Face username
   - **Password:** the **write** token

   (Or via CLI: `pip install -U huggingface_hub`, `hf auth login`,
   `hf repo create cathay-pacific-sentiment-analysis --repo-type space --space-sdk docker`.)

4. The Space builds the Docker image (a few minutes; first run downloads
   Whisper-small) and goes live at the Space URL.

---

## Notes

- **No secrets required at runtime.** The model
  `tonyho5689/cathay-pacific-sentiment-analysis` is public on the Hub.
- **Sleep:** free Spaces pause after 48h idle and wake on the next visit. To never
  pause, upgrade the Space hardware (Settings → Hardware) to a paid tier.

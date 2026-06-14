# Deploying to Hugging Face Spaces

This app is configured to run on **Hugging Face Spaces** (Streamlit SDK). Spaces
is a better home than Streamlit Community Cloud for this project because:

- Your fine-tuned model already lives on the HF Hub.
- Free CPU Spaces have **16 GB RAM / 2 vCPU** — plenty for Whisper-small + the
  sentiment model.
- A free Space only **pauses after 48 hours of zero traffic** (and wakes on the
  next visit) — far gentler than Streamlit Cloud's aggressive sleep. Upgrading to
  paid hardware removes the pause entirely.

Everything needed is already in this repo:

| File | Purpose |
|------|---------|
| `README.md` (YAML frontmatter) | Tells Spaces this is a Streamlit app (`app_file: app.py`) |
| `app.py` | The Streamlit application |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System packages (`ffmpeg`, for audio decoding) |
| `.streamlit/config.toml` | Theme + 50 MB upload limit |

---

## Option A — Create the Space and push with git (recommended)

1. **Create the Space**
   - Go to <https://huggingface.co/new-space>.
   - **Owner:** your account · **Space name:** e.g. `cathay-pacific-sentiment-analysis`.
   - **License:** your choice (e.g. MIT).
   - **SDK:** select **Streamlit**.
   - **Hardware:** **CPU basic** (free).
   - **Visibility:** Public.
   - Click **Create Space**.

2. **Get a write token**
   - <https://huggingface.co/settings/tokens> → **New token** → role **Write** → copy it.

3. **Push this repo to the Space**

   From the project folder:

   ```bash
   # Add the Space as a git remote (replace <user> and <space-name>)
   git remote add space https://huggingface.co/spaces/<user>/<space-name>

   # Push the current main branch to the Space's main branch
   git push space main
   ```

   When prompted for credentials:
   - **Username:** your Hugging Face username
   - **Password:** paste the **write token** from step 2 (not your account password)

4. **Wait for the build**
   - The Space auto-builds (installs `packages.txt` then `requirements.txt`).
   - First build takes a few minutes (it downloads Whisper-small on first run).
   - Your app goes live at `https://huggingface.co/spaces/<user>/<space-name>`.

---

## Option B — Upload files in the browser (no git)

1. Create the Space as in Option A, step 1.
2. Open the Space → **Files** tab → **Add file → Upload files**.
3. Upload: `app.py`, `requirements.txt`, `packages.txt`, `README.md`, and the
   `.streamlit/config.toml` (keep it inside a `.streamlit/` folder).
4. The Space rebuilds automatically after each upload.

---

## Notes

- **No secrets required.** The model `tonyho5689/cathay-pacific-sentiment-analysis`
  is public on the Hub, so inference needs no token.
- **`sdk_version`** in the README frontmatter is pinned to a known-good Streamlit
  version. If the build complains it's unavailable, bump it to a version listed in
  the Space's build logs (or remove the line to let Spaces pick the latest).
- **Sleep:** free Spaces pause after 48h idle and wake on the next visit. To never
  pause, upgrade the Space hardware (Settings → Hardware) to a paid tier.
- **Keeping GitHub and the Space in sync:** re-run `git push space main` after each
  change, or set up the optional GitHub Action that mirrors pushes to the Space
  (ask if you want this added — it needs an `HF_TOKEN` repo secret).

# 🎙️ Meeting Transcription & Summarization

A Streamlit app that turns a meeting recording into a **transcript, summary, action items and keyword insights**.
Upload an audio file (WAV, MP3, M4A, MPEG, MPGA) or record straight from your microphone.

| Step | Tool |
|------|------|
| Speech → text | [OpenAI Whisper](https://github.com/openai/whisper) (`base` or `small`) |
| Summarization | [DistilBART](https://huggingface.co/sshleifer/distilbart-cnn-12-6) (`distilbart-cnn-12-6`) |
| Action items | Keyword matching (*need to, should, will, deadline, …*) |

**Features:** timestamped + searchable transcript, summary, tick-off action items, top-keyword chart,
speaking stats, and `.txt` export (transcript / summary / full report).

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
sudo apt install ffmpeg        # macOS: brew install ffmpeg · Windows: choco install ffmpeg
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first run downloads the models (Whisper `base` ≈ 140 MB, DistilBART ≈ 1.2 GB).
Try it with `sample_audio/meeting.m4a`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io> → **Create app** → pick the repo, branch `main`, main file `streamlit_app.py`.
3. Under **Advanced settings** choose **Python 3.11**.
4. Deploy. `requirements.txt` (Python packages) and `packages.txt` (`ffmpeg`) are picked up automatically.

> ⚠️ **Memory:** Community Cloud apps have roughly 2.7 GB RAM. `base` Whisper + DistilBART fits, but the
> `small` Whisper option may run out of memory, and the first request is slow while models download.
> If you hit limits, deploy to a [Hugging Face Space](https://huggingface.co/spaces) (Streamlit SDK) with more RAM instead.

## Repo layout

```
streamlit_app.py        # the app (entry point)
requirements.txt        # Python dependencies (CPU-only PyTorch)
packages.txt            # system packages for Streamlit Cloud (ffmpeg)
.streamlit/config.toml  # dark theme
notebooks/              # original exploration notebook (Whisper + BART pipeline)
sample_audio/           # short test recording
```

## Notes

- Summaries are truncated to DistilBART's input limit, so long meetings are summarized from roughly the first part only — check the transcript for full detail.
- Action-item detection is keyword-based and will include false positives.
- The full AMI corpus recording used in the notebook (`ES2002a.Mix-Headset.wav`, ~40 MB) is intentionally not in the repo.

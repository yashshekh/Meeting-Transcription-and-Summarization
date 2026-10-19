import os
os.environ["USE_TF"] = "0"

import re
import tempfile
from collections import Counter

import pandas as pd
import streamlit as st
import whisper
from transformers import pipeline

st.set_page_config(
    page_title="Meeting summarizer",
    page_icon=":material/graphic_eq:",
    layout="wide",
)

ACTION_KEYWORDS = [
    "need to", "should", "will", "must",
    "action", "follow up", "next step",
    "responsible for", "deadline", "assign",
]

STOPWORDS = set("""
a an the and or but if so of to in on at for with by from as is are was were be been
being it its this that these those i you he she we they me him her us them my your our
their what which who whom do does did doing have has had having not no yes just very can
could would should will shall may might must also than then there here about into over
up down out off again further once all any both each few more most other some such only
own same too s t don now um uh okay ok yeah right well like know think going get got
really actually mean one oh so gonna want let
""".split())


LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}


@st.cache_resource
def load_whisper(size):
    return whisper.load_model(size)


@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        framework="pt",
    )


def analyze(audio_bytes, suffix, status, model_size, language):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        status.update(label=f"Loading Whisper ({model_size}) and BART models...")
        whisper_model = load_whisper(model_size)
        summarizer = load_summarizer()

        # Decode once so the duration comes from the real audio length, not
        # from Whisper's segment timestamps (which can overshoot on short clips).
        audio = whisper.load_audio(temp_path)
        duration = len(audio) / whisper.audio.SAMPLE_RATE

        status.update(label="Transcribing meeting audio...")
        result = whisper_model.transcribe(audio, fp16=False, language=language)
        transcript = result["text"].strip()
        if not transcript:
            return None

        status.update(label="Generating meeting summary...")
        summary_result = summarizer(
            transcript,
            max_length=150,
            min_length=40,
            do_sample=False,
            truncation=True,
        )
        meeting_summary = summary_result[0]["summary_text"]

        sentences = re.split(r"(?<=[.!?])\s+", transcript)
        action_items = [
            sentence.strip()
            for sentence in sentences
            if any(word in sentence.lower() for word in ACTION_KEYWORDS)
        ]

        segments = [
            {
                "Start": round(min(seg["start"], duration), 1),
                "End": round(min(seg["end"], duration), 1),
                "Text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ]

        return {
            "transcript": transcript,
            "summary": meeting_summary,
            "action_items": action_items,
            "segments": segments,
            "duration": duration,
            "language": result.get("language", language or "unknown"),
            "model_size": model_size,
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def format_time(seconds):
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def top_keywords(text, n=10):
    words = [
        w for w in re.findall(r"[a-z']+", text.lower())
        if len(w) > 2 and w not in STOPWORDS
    ]
    return Counter(words).most_common(n)


def build_report(res):
    return (
        "MEETING SUMMARY\n"
        + res["summary"]
        + "\n\nFULL TRANSCRIPT\n"
        + res["transcript"]
        + "\n\nPOSSIBLE ACTION ITEMS\n"
        + "\n".join("- " + item for item in res["action_items"])
    )


if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.file_id = None
    st.session_state.input_nonce = 0


def reset_audio():
    # New widget keys give a fresh, empty recorder/uploader.
    st.session_state.input_nonce += 1
    st.session_state.results = None
    st.session_state.file_id = None
    for key in [k for k in st.session_state if str(k).startswith("action_")]:
        del st.session_state[key]

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("How it works", icon=":material/account_tree:")
    st.markdown(
        ":material/mic: **1. Speech to text**  \n"
        ":gray[OpenAI Whisper `base`]\n\n"
        ":material/summarize: **2. Summarization**  \n"
        ":gray[DistilBART `cnn-12-6`]\n\n"
        ":material/checklist: **3. Action items**  \n"
        ":gray[Keyword-based extraction]"
    )
    st.space("small")
    st.subheader("Settings", icon=":material/tune:")
    language_label = st.selectbox(
        "Spoken language",
        list(LANGUAGES),
        index=1,
        help="Setting the language avoids Whisper guessing wrong on short "
             "or accented recordings.",
    )
    model_size = st.segmented_control(
        "Whisper model",
        ["base", "small"],
        default="base",
        required=True,
        help="`small` is noticeably more accurate, especially with accents, "
             "but slower and downloads ~460 MB on first use.",
    )
    st.space("small")
    st.caption(
        "Upload WAV, MP3, M4A, MPEG or MPGA files, or record directly "
        "from your microphone. "
        "Models run locally on your machine; the first run downloads them."
    )

# ---------- Header ----------
st.title("Meeting transcription and summarization", icon=":material/graphic_eq:")
st.caption(
    "Upload a meeting recording or record your voice to get a full transcript, a concise summary, "
    "keyword insights and possible action items."
)
st.markdown(
    ":violet-badge[:material/mic: Whisper] "
    ":blue-badge[:material/auto_awesome: BART] "
    ":green-badge[:material/lock: Runs locally]"
)

# ---------- Upload ----------
UPLOAD = ":material/upload_file: Upload a file"
RECORD = ":material/mic: Record your voice"

with st.container(border=True):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        source = st.segmented_control(
            "Audio source",
            [UPLOAD, RECORD],
            default=UPLOAD,
            required=True,
            key="audio_source",
        )
        st.button(
            "Refresh",
            icon=":material/refresh:",
            on_click=reset_audio,
            help="Clear the current recording or file and its results.",
        )
    nonce = st.session_state.input_nonce

    left, right = st.columns([1, 1], vertical_alignment="center")
    with left:
        if source == RECORD:
            audio_file = st.audio_input(
                "Record your voice",
                key=f"recorder_{nonce}",
                help="Click the microphone to start, click again to stop. "
                     "Your browser will ask for microphone permission the first time.",
            )
            audio_name = "voice recording"
            audio_suffix = ".wav"
        else:
            audio_file = st.file_uploader(
                "Upload meeting audio",
                type=["wav", "mp3", "m4a", "mpeg", "mpga"],
                key=f"uploader_{nonce}",
            )
            audio_name = audio_file.name if audio_file else ""
            audio_suffix = os.path.splitext(audio_name)[1]
    with right:
        if audio_file is not None:
            st.caption(f":material/audio_file: **{audio_name}** · "
                       f"{audio_file.size / 1024:,.1f} KB")
            st.audio(audio_file, alt=f"Audio player for {audio_name}")
        elif source == RECORD:
            st.caption(":material/info: Record something, then play it back here.")
        else:
            st.caption(":material/info: Choose a recording to preview it here.")

    run = st.button(
        "Transcribe and summarize",
        type="primary",
        icon=":material/play_arrow:",
        disabled=audio_file is None,
        width="stretch",
    )

if audio_file is None:
    st.session_state.results = None
    st.session_state.file_id = None
elif audio_file.file_id != st.session_state.file_id:
    st.session_state.results = None

if run and audio_file is not None:
    with st.status("Analyzing meeting...", expanded=False) as status:
        try:
            res = analyze(
                audio_file.getvalue(),
                audio_suffix,
                status,
                model_size,
                LANGUAGES[language_label],
            )
        except Exception as error:
            status.update(label="Analysis failed", state="error")
            st.error(f"An error occurred: {error}")
            st.stop()

        if res is None:
            status.update(label="No speech detected", state="error")
            st.warning("No speech could be transcribed.")
            st.stop()

        status.update(label="Meeting analysis completed", state="complete")
    st.session_state.results = res
    st.session_state.file_id = audio_file.file_id
    st.toast("Meeting analysis completed!", icon=":material/check_circle:")

res = st.session_state.results
if res is None:
    st.stop()

# ---------- Metrics ----------
transcript_words = len(res["transcript"].split())
summary_words = len(res["summary"].split())
reduction = (1 - summary_words / transcript_words) * 100 if transcript_words else 0

with st.container(horizontal=True):
    st.metric("Duration", format_time(res["duration"]), icon=":material/schedule:", border=True)
    st.metric("Transcript words", f"{transcript_words:,}", icon=":material/notes:", border=True)
    st.metric("Summary words", f"{summary_words:,}", icon=":material/short_text:", border=True)
    st.metric("Text reduction", f"{max(reduction, 0):.0f}%", icon=":material/compress:", border=True,
              help="How much shorter the summary is than the transcript.")
    st.metric("Action items", len(res["action_items"]), icon=":material/task_alt:", border=True)

st.caption(
    f":material/translate: Language: **{res['language']}** · "
    f":material/memory: Whisper model: **{res['model_size']}**"
)

# ---------- Results ----------
tab_summary, tab_transcript, tab_actions, tab_insights, tab_export = st.tabs([
    ":material/summarize: Summary",
    ":material/description: Transcript",
    ":material/checklist: Action items",
    ":material/insights: Insights",
    ":material/download: Export",
])

with tab_summary:
    with st.container(border=True):
        st.subheader("Meeting summary", icon=":material/auto_awesome:")
        st.markdown(res["summary"])
    st.caption(
        "Generated by DistilBART. Long meetings are truncated to the model's "
        "input limit, so check the transcript for full details."
    )

with tab_transcript:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        view = st.segmented_control(
            "View",
            [":material/schedule: Timestamped", ":material/article: Plain text"],
            default=":material/schedule: Timestamped",
            key="transcript_view",
        )
        query = st.text_input(
            "Search transcript",
            placeholder="Search for a word or phrase...",
            icon=":material/search:",
        )

    if view and "Plain" in view:
        text = res["transcript"]
        if query:
            text = re.sub(
                f"({re.escape(query)})", r"**:orange[\1]**", text, flags=re.IGNORECASE
            )
        with st.container(border=True, height=420):
            st.markdown(text)
    else:
        df = pd.DataFrame(res["segments"])
        if query and not df.empty:
            df = df[df["Text"].str.contains(query, case=False, regex=False)]
        st.caption(f"{len(df)} segment(s)" + (f" matching “{query}”" if query else ""))
        st.dataframe(
            df,
            hide_index=True,
            height=420,
            column_config={
                "Start": st.column_config.NumberColumn("Start (s)", format="%.1f", width="small"),
                "End": st.column_config.NumberColumn("End (s)", format="%.1f", width="small"),
                "Text": st.column_config.TextColumn("Spoken text", width="large"),
            },
            alt="Transcript segments with start and end times",
        )

with tab_actions:
    items = res["action_items"]
    if not items:
        st.info(
            "No possible action items were identified by the keyword-based method.",
            icon=":material/info:",
        )
    else:
        done = sum(st.session_state.get(f"action_{i}", False) for i in range(len(items)))
        st.progress(done / len(items), text=f"{done} of {len(items)} items marked done")
        with st.container(border=True):
            for i, item in enumerate(items):
                st.checkbox(item, key=f"action_{i}")
        st.caption(
            "Detected by keyword matching (e.g. *need to*, *should*, *will*, *deadline*). "
            "Tick items off as you review them."
        )

with tab_insights:
    keywords = top_keywords(res["transcript"])
    col_chart, col_info = st.columns([2, 1])
    with col_chart:
        with st.container(border=True):
            st.subheader("Most frequent topics", icon=":material/bar_chart:")
            if keywords:
                kw_df = pd.DataFrame(keywords, columns=["Word", "Count"])
                st.bar_chart(
                    kw_df, x="Word", y="Count", horizontal=True, sort="-Count",
                    height=320, alt="Bar chart of the most frequent words in the meeting",
                )
            else:
                st.caption("Not enough text to find recurring topics.")
    with col_info:
        with st.container(border=True, height="stretch"):
            st.subheader("Speaking stats", icon=":material/record_voice_over:")
            minutes = res["duration"] / 60 if res["duration"] else 0
            st.metric("Words per minute", f"{transcript_words / minutes:.0f}" if minutes else "–")
            st.metric("Segments", len(res["segments"]))
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", res["transcript"]) if s]
            st.metric("Sentences", len(sentences))

with tab_export:
    st.caption("Download the results as plain text files.")
    with st.container(horizontal=True):
        st.download_button(
            "Transcript", res["transcript"], file_name="transcript.txt",
            mime="text/plain", icon=":material/description:", on_click="ignore",
        )
        st.download_button(
            "Summary", res["summary"], file_name="meeting_summary.txt",
            mime="text/plain", icon=":material/summarize:", on_click="ignore",
        )
        st.download_button(
            "Full meeting report", build_report(res), file_name="meeting_report.txt",
            mime="text/plain", icon=":material/download:", type="primary", on_click="ignore",
        )
    with st.expander("Preview report", icon=":material/visibility:"):
        st.code(build_report(res), language=None, wrap_lines=True)

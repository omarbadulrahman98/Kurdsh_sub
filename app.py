import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
import yt_dlp
from deep_translator import GoogleTranslator

# ───────────────────────── ڕێکخستنی گشتی ─────────────────────────

st.set_page_config(page_title="ژێرنووسی کوردی", page_icon="🎬", layout="centered")

CLAUDE_MODEL = "claude-sonnet-5-5"
CHUNK_SIZE = 30          # ژمارەی دێڕەکان لە هەر داواکارییەکی Claude
MAX_VIDEOS = 10          # زۆرترین ژمارەی ڤیدیۆ لە یەک جار
LINE_WIDTH = 42          # درێژی دێڕی ژێرنووس

LANGS = {
    "فارسی": "fa",
    "عەرەبی": "ar",
    "ئینگلیزی": "en",
    "تورکی": "tr",
    "دیاریکردنی خۆکار": None,
}
LANG_NAMES = {"fa": "Persian", "ar": "Arabic", "en": "English", "tr": "Turkish", None: "the source language"}

BUILTIN_FIXES = {
    "ئیران": "ئێران",
    "ئيران": "ئێران",
}

OUT_DIR = os.path.join(tempfile.gettempdir(), "kurdish_subs_out")
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
os.makedirs(OUT_DIR, exist_ok=True)

YOUTUBE_RE = re.compile(r"^https?://(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/", re.I)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600;800&display=swap');
html, body, .stApp, .stMarkdown, p, label, h1, h2, h3, h4,
.stButton button, .stDownloadButton button, .stTextArea textarea, .stTextInput input {
    font-family: 'Vazirmatn', 'Segoe UI', Tahoma, sans-serif;
}
.stApp { direction: rtl; background: #f6f7f4; }
[data-testid="stSidebar"] { direction: rtl; background: #ecefe9; }
.stTextArea textarea, .stTextInput input { direction: ltr; text-align: left; }
.block-container { max-width: 760px; padding-top: 2.2rem; }
.hero { border-right: 6px solid #1f6f5c; padding: 0.4rem 1.2rem; margin-bottom: 1.6rem; }
.hero h1 { font-weight: 800; font-size: 2.1rem; margin: 0 0 .4rem 0; color: #16302a; }
.hero p { margin: 0; color: #46524d; font-size: 1.02rem; line-height: 1.9; }
.stButton button[kind="primary"] { background: #1f6f5c; border: none; font-weight: 600; }
.stButton button[kind="primary"]:hover { background: #17564a; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────────────────────── فەنکشنە یاریدەدەرەکان ─────────────────────────


def format_time(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_ku(text: str, extra_fixes: dict) -> str:
    text = (
        text.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("ى", "ی")
        .replace("ـ", "")
    )
    for bad, good in {**BUILTIN_FIXES, **extra_fixes}.items():
        if bad:
            text = text.replace(bad, good)
    return re.sub(r"[ \t]+", " ", text).strip()


def build_srt(df: pd.DataFrame, column: str) -> str:
    blocks = []
    n = 0
    for _, row in df.iterrows():
        text = str(row[column] or "").strip()
        if not text:
            continue
        n += 1
        start = float(row["start"])
        end = max(float(row["end"]), start + 0.3)
        wrapped = "\n".join(textwrap.wrap(text, LINE_WIDTH)) or text
        blocks.append(f"{n}\n{format_time(start)} --> {format_time(end)}\n{wrapped}\n")
    return "\n".join(blocks)


def build_txt(df: pd.DataFrame, column: str) -> str:
    return "\n".join(str(t).strip() for t in df[column] if str(t).strip())


def safe_name(title: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", title, flags=re.UNICODE).strip("_")
    return (name or "video")[:60]


def parse_glossary(text: str):
    """هەر دێڕێک: وشەی فارسی=وشەی کوردی"""
    pairs = []
    for line in text.splitlines():
        if "=" in line:
            src, dst = line.split("=", 1)
            if src.strip() and dst.strip():
                pairs.append((src.strip(), dst.strip()))
    return pairs


def get_api_key(manual: str) -> str:
    if manual.strip():
        return manual.strip()
    try:
        key = st.secrets["ANTHROPIC_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")


def whisper_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


@st.cache_resource(show_spinner=False)
def get_model(name: str):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="auto", compute_type="int8")


# ───────────────────────── وەرگێڕان ─────────────────────────


TRANSLATE_STATS = {"failed": 0, "last_error": ""}


def translate_google(texts, src_code):
    source = src_code or "auto"

    def one(text):
        if not str(text).strip():
            return text
        for attempt in range(4):
            try:
                result = GoogleTranslator(source=source, target="ckb").translate(text)
                if result:
                    return result
            except Exception as e:
                TRANSLATE_STATS["last_error"] = f"{type(e).__name__}: {e}"[:300]
                time.sleep(1 + attempt)
        TRANSLATE_STATS["failed"] += 1
        return text

    with ThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(one, texts))


SYSTEM_PROMPT = (
    "You are a professional subtitle translator into Central Kurdish (Sorani), written in Arabic script. "
    "Rules: translate the meaning naturally as spoken language, in short subtitle style. "
    "Use correct Sorani orthography (ێ ۆ ڕ ڵ ە). Write Iran as ئێران and the preposition 'in' as لە. "
    "Transliterate personal names and places into Sorani script. Keep numbers. "
    "Do not merge, split, add or drop lines. Do not explain anything. "
    "Output only a JSON object."
)


def parse_json_object(raw: str) -> dict:
    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object")
    return json.loads(raw[start : end + 1])


def translate_claude(client, texts, title, src_code, glossary, on_progress):
    results = [None] * len(texts)
    gloss_text = "\n".join(f"{a} = {b}" for a, b in glossary)
    total_chunks = max(1, (len(texts) + CHUNK_SIZE - 1) // CHUNK_SIZE)

    for c, start in enumerate(range(0, len(texts), CHUNK_SIZE)):
        idx = list(range(start, min(start + CHUNK_SIZE, len(texts))))
        payload = {str(i): texts[i] for i in idx}
        prompt = (
            f"Video title (context): {title}\n"
            f"Source language: {LANG_NAMES.get(src_code, 'the source language')}\n"
            + (f"Glossary (always use these translations):\n{gloss_text}\n" if gloss_text else "")
            + "\nTranslate every value of this JSON object into Central Kurdish (Sorani). "
            "Return a JSON object with exactly the same keys.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=8000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                data = parse_json_object(raw)
                for i in idx:
                    val = data.get(str(i))
                    if isinstance(val, str) and val.strip():
                        results[i] = val.strip()
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        on_progress((c + 1) / total_chunks)

    missing = [i for i, r in enumerate(results) if r is None]
    if missing:
        fallback = translate_google([texts[i] for i in missing], src_code)
        for i, t in zip(missing, fallback):
            results[i] = t
    return results


# ───────────────────────── پرۆسەی ڤیدیۆ ─────────────────────────


def _match_track(tracks: dict, code: str):
    for key in tracks:
        if key == code or key.startswith(code + "-"):
            return key
    return None


def pick_caption_track(info: dict, src_code):
    """ڕاسپاردەی باشترین ژێرنووس: یەکەم دەستی، ئینجا خۆکاریی ڕەسەنی یوتوب (نەک وەرگێڕدراو)."""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    codes = [src_code] if src_code else [info.get("language"), "fa", "ar", "en", "tr"]
    codes = [c for c in codes if c]
    has_orig = any(k.endswith("-orig") for k in auto)
    for code in codes:
        key = _match_track(manual, code)
        if key:
            return key, True
        if has_orig:
            if f"{code}-orig" in auto:
                return f"{code}-orig", False
        else:
            key = _match_track(auto, code)
            if key:
                return key, False
    return None, False


def parse_json3(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs or "tStartMs" not in ev:
            continue
        text = "".join(sg.get("utf8", "") for sg in segs).replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if not text or re.fullmatch(r"[\[\(][^\]\)]*[\]\)]|[♪\s]+", text):
            continue
        start = ev["tStartMs"] / 1000
        end = start + ev.get("dDurationMs", 2000) / 1000
        if rows and rows[-1]["original"] == text:
            rows[-1]["end"] = max(rows[-1]["end"], end)
            continue
        rows.append({"start": start, "end": end, "original": text})
    rows.sort(key=lambda r: r["start"])
    for cur, nxt in zip(rows, rows[1:]):
        if cur["end"] > nxt["start"]:
            cur["end"] = max(cur["start"] + 0.3, nxt["start"])
    return rows


def fetch_youtube_captions(url, info, src_code, cookies_path):
    key, is_manual = pick_caption_track(info, src_code)
    if not key:
        return None, ""
    with tempfile.TemporaryDirectory() as tmp:
        opts = {
            **ydl_base_opts(cookies_path),
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [key],
            "subtitlesformat": "json3",
            "outtmpl": os.path.join(tmp, "subs.%(ext)s"),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        files = [f for f in os.listdir(tmp) if f.endswith(".json3")]
        if not files:
            return None, ""
        rows = parse_json3(os.path.join(tmp, files[0]))
    label = "ژێرنووسی دەستیی ڤیدیۆ" if is_manual else "ژێرنووسی خۆکاریی یوتوب"
    return (rows or None), label


def transcribe_whisper(url, duration, opts, bar):
    if not whisper_available():
        raise RuntimeError(
            "ژێرنووسی یوتوب نەدۆزرایەوە و faster-whisper دانەمەزراوە. "
            "لە requirements.txt هێمای # لە پێش faster-whisper لابەرە."
        )
    with tempfile.TemporaryDirectory() as tmp:
        bar.progress(0.05, text="خەریکی داونلۆدکردنی دەنگەکەم...")
        dl_opts = {
            **ydl_base_opts(opts["cookies_path"]),
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmp, "audio.%(ext)s"),
        }
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([url])
        files = [f for f in os.listdir(tmp) if f.startswith("audio.")]
        if not files:
            raise RuntimeError("داونلۆدکردنی دەنگ سەرکەوتوو نەبوو.")
        audio_path = os.path.join(tmp, files[0])

        bar.progress(0.15, text="خەریکی گۆڕینی دەنگ بۆ دەقم...")
        model = get_model(opts["whisper_model"])
        segments, tinfo = model.transcribe(
            audio_path,
            language=opts["src_code"],
            task="transcribe",
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
        )
        total = tinfo.duration or duration
        rows = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                rows.append({"start": seg.start, "end": seg.end, "original": text})
            bar.progress(0.15 + 0.55 * min(seg.end / total, 1.0), text="خەریکی گۆڕینی دەنگ بۆ دەقم...")
    return rows


def translate_rows(rows, title, opts, status, bar):
    texts = [r["original"] for r in rows]
    TRANSLATE_STATS["failed"] = 0
    TRANSLATE_STATS["last_error"] = ""
    glossary = parse_glossary(opts["glossary"])
    extra_fixes = {}
    bar.progress(0.72, text="خەریکی وەرگێڕانم بۆ کوردی سۆرانی...")

    def on_progress(frac):
        bar.progress(0.72 + 0.26 * frac, text="خەریکی وەرگێڕانم بۆ کوردی سۆرانی...")

    client = None
    if opts["engine"] == "claude" and opts["api_key"]:
        import anthropic

        client = anthropic.Anthropic(api_key=opts["api_key"])
        translated = translate_claude(client, texts, title, opts["src_code"], glossary, on_progress)
    else:
        translated = translate_google(texts, opts["src_code"])
        on_progress(1.0)

    failed = TRANSLATE_STATS["failed"]
    if failed:
        err = TRANSLATE_STATS["last_error"] or "نەزانراو"
        if failed >= len(texts):
            raise RuntimeError(
                "وەرگێڕان بۆ کوردی سەرکەوتوو نەبوو (گووگڵ ڕێگری کرد یان وەڵامی نەدایەوە)، "
                "بۆیە دەقەکە هەر بە زمانی ڕەسەن دەمایەوە. دووبارە هەوڵبدەرەوە.\n\n"
                f"هۆکار: {err}"
            )
        status.write(f"⚠️ {failed} دێڕ وەرنەگێڕدرا و بە زمانی ڕەسەن ماونەتەوە. هۆکار: {err}")

    for r, t in zip(rows, translated):
        r["kurdish"] = clean_ku(t, extra_fixes)

    ku_title = ""
    try:
        if client:
            ku_title = translate_claude(client, [title], title, opts["src_code"], glossary, lambda f: None)[0]
        else:
            ku_title = translate_google([title], opts["src_code"])[0]
        ku_title = clean_ku(ku_title, extra_fixes)
    except Exception:
        ku_title = ""

    df = pd.DataFrame(rows, columns=["start", "end", "original", "kurdish"])
    return df, ku_title


def process_video(url, opts, status, bar):
    ydl_base = ydl_base_opts(opts["cookies_path"])

    bar.progress(0.02, text="خەریکی خوێندنەوەی زانیاری ڤیدیۆم...")
    with yt_dlp.YoutubeDL(ydl_base) as ydl:
        info = ydl.extract_info(url, download=False)
    title = info.get("title") or "video"
    duration = info.get("duration")
    vid = info.get("id") or safe_name(title)
    if not duration or info.get("is_live"):
        raise RuntimeError("ئەم ڤیدیۆیە لایڤە یان کاتەکەی نەزانراوە.")
    if duration > opts["max_minutes"] * 60:
        raise RuntimeError(
            f"ڤیدیۆکە {duration // 60} خولەکە، و سنووری ئێستا {opts['max_minutes']} خولەکە. لە لایەنی لاوە سنوورەکە زیاد بکە."
        )
    status.write(f"**{title}** — {duration // 60} خولەک")

    # ١. دەقی ڕەسەن: ژێرنووسی یوتوب (خێرا) یان Whisper
    rows = None
    if opts["mode"] == "captions":
        bar.progress(0.1, text="خەریکی وەرگرتنی ژێرنووسی یوتوبم...")
        try:
            rows, source = fetch_youtube_captions(url, info, opts["src_code"], opts["cookies_path"])
        except Exception:
            rows, source = None, ""
        if rows:
            status.write(f"سەرچاوەی دەق: {source}")
        elif not opts["fallback_whisper"]:
            raise RuntimeError(
                "ژێرنووسی یوتوب بۆ ئەم ڤیدیۆیە نەدۆزرایەوە (یان یوتوب ڕێگری کرد). "
                "لە ڕێکخستنەکان Whisper وەک پاشەکشە چالاک بکە."
            )
        else:
            status.write("ژێرنووسی یوتوب نەدۆزرایەوە، Whisper بەکاردێت...")
    if not rows:
        rows = transcribe_whisper(url, duration, opts, bar)
    if not rows:
        raise RuntimeError("هیچ قسەیەک لە ڤیدیۆکەدا نەدۆزرایەوە.")

    df, ku_title = translate_rows(rows, title, opts, status, bar)
    bar.progress(1.0, text="تەواو بوو")
    return {
        "id": vid, "url": url, "title": title, "ku_title": ku_title,
        "duration": int(duration), "df": df, "rev": int(time.time()),
    }



# ───────────────────────── دروستکردنی ڤیدیۆی ژێرنووسکراو ─────────────────────────


def process_upload(uploaded, opts, status, bar):
    """ڤیدیۆی ئەپلۆدکراو: دەنگ ← Whisper ← وەرگێڕان (بێ یوتوب)."""
    if not whisper_available():
        raise RuntimeError(
            "faster-whisper دانەمەزراوە. لە requirements.txt دڵنیابە ئەم دێڕە هەیە: faster-whisper>=1.0.3"
        )
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg دانەمەزراوە. لە packages.txt دایبنێ.")

    bar.progress(0.02, text="خەریکی پاشەکەوتکردنی ڤیدیۆکەم...")
    ext = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
    vid = f"up{int(time.time())}"
    local_path = os.path.join(OUT_DIR, f"{vid}{ext}")
    with open(local_path, "wb") as f:
        f.write(uploaded.getbuffer())

    duration = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", local_path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )
    if duration > opts["max_minutes"] * 60:
        raise RuntimeError(f"ڤیدیۆکە {int(duration // 60)} خولەکە و سنووری ئێستا {opts['max_minutes']} خولەکە.")
    title = os.path.splitext(uploaded.name)[0]
    status.write(f"**{title}** — {int(duration // 60)} خولەک")

    bar.progress(0.06, text="خەریکی دەرهێنانی دەنگم...")
    # دەنگ ڕاستەوخۆ بە ffmpeg دەخوێنینەوە (بێ پێویستی بە av ی faster-whisper)
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", local_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
    )
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(
            "دەنگ لەم ڤیدیۆیە دەرنەهات. دڵنیابە فایلەکە دەنگی تێدایە.\n" + r.stderr.decode("utf-8", "ignore")[-300:]
        )
    import numpy as np

    audio = np.frombuffer(r.stdout, dtype=np.float32)

    bar.progress(0.12, text="خەریکی گۆڕینی دەنگ بۆ دەقم (ئەمە کەمێک کات دەخایەنێت)...")
    model = get_model(opts["whisper_model"])
    segments, tinfo = model.transcribe(
        audio,
        language=opts["src_code"],
        task="transcribe",
        vad_filter=True,
        beam_size=opts.get("beam_size", 5),
        condition_on_previous_text=False,
    )
    total = duration
    rows = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            rows.append({"start": seg.start, "end": seg.end, "original": text})
        bar.progress(0.12 + 0.58 * min(seg.end / total, 1.0), text="خەریکی گۆڕینی دەنگ بۆ دەقم...")
    if not rows:
        raise RuntimeError("هیچ قسەیەک لە ڤیدیۆکەدا نەدۆزرایەوە.")
    status.write(f"زمانی دۆزراوە: {getattr(tinfo, 'language', '?')}")

    df, _ = translate_rows(rows, title, opts, status, bar)
    ku_title = ""  # ناوی فایل سەرناوی ڕاستەقینە نییە؛ خۆت لە خانەی سەرناو بینووسە
    bar.progress(1.0, text="تەواو بوو")
    return {
        "id": vid, "url": "", "local_path": local_path, "title": title, "ku_title": ku_title,
        "duration": int(duration), "df": df, "rev": int(time.time()),
    }


def ydl_base_opts(cookies_path: str) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "retries": 3}
    if cookies_path:
        opts["cookiefile"] = cookies_path
    return opts


def write_cookies(upload) -> str:
    if upload is None:
        return ""
    ck = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    ck.write(upload.getvalue())
    ck.close()
    return ck.name


def format_ass_time(seconds: float) -> str:
    cs_total = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs_total, 360_000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    text = text.replace("{", "(").replace("}", ")").replace("\\", "/")
    text = " ".join(text.split("\n")).strip()
    # ئاراستەی ڕاستەچەپ بۆ libass (خاڵ و نیشانەکان لە لای چەپ بن)
    return "\u202b" + text + "\u202c"


def build_ass(df, title_text, font, cw, ch, vertical, video_top, video_bottom, total_seconds) -> str:
    if vertical:
        sub_size = int(cw * 0.061)
        sub_align = 8                       # لە سەرەوەی ناوچەی ژێرەوە دەست پێدەکات
        sub_margin_v = video_bottom + int(ch * 0.025)
        title_size = int(cw * 0.058)
        title_margin_v = max(40, int(video_top * 0.30))
    else:
        sub_size = int(ch * 0.05)
        sub_align = 2                       # ژێر ڤیدیۆکە
        sub_margin_v = int(ch * 0.06)
        title_size = int(ch * 0.05)
        title_margin_v = int(ch * 0.05)
    side = int(cw * 0.06)
    outline = max(2, int(cw * 0.003))

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {cw}",
        f"PlayResY: {ch}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Sub,{font},{sub_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},0,{sub_align},{side},{side},{sub_margin_v},1",
        f"Style: Title,{font},{title_size},&H0000FFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},0,8,{side},{side},{title_margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if title_text.strip():
        lines.append(
            f"Dialogue: 0,{format_ass_time(0)},{format_ass_time(total_seconds + 1)},Title,,0,0,0,,{ass_escape(title_text)}"
        )
    for _, row in df.iterrows():
        text = str(row["kurdish"] or "").strip()
        if not text:
            continue
        start = float(row["start"])
        end = max(float(row["end"]), start + 0.3)
        lines.append(
            f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},Sub,,0,0,0,,{ass_escape(text)}"
        )
    return "\n".join(lines) + "\n"


def probe_size(path: str):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def render_video(url, df, title_text, opts, bar, upload=None, local_path=None):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg دانەمەزراوە. لە packages.txt یان بە apt install ffmpeg دایبمەزرێنە.")

    with tempfile.TemporaryDirectory() as tmp:
        if local_path and os.path.exists(local_path):
            bar.progress(0.02, text="خەریکی ئامادەکردنی ڤیدیۆکەم...")
            src_path = os.path.join(tmp, "video" + (os.path.splitext(local_path)[1] or ".mp4"))
            shutil.copy(local_path, src_path)
        elif upload is not None:
            # ڤیدیۆی ئەپلۆدکراو: پێویست بە داونلۆد لە یوتوب نییە
            bar.progress(0.02, text="خەریکی ئامادەکردنی ڤیدیۆی ئەپلۆدکراوم...")
            ext = os.path.splitext(upload.name)[1].lower() or ".mp4"
            src_path = os.path.join(tmp, "video" + ext)
            with open(src_path, "wb") as f:
                f.write(upload.getbuffer())
        else:
            bar.progress(0.02, text="خەریکی داونلۆدکردنی ڤیدیۆم...")
            dl_opts = {
                **ydl_base_opts(opts["cookies_path"]),
                "format": "bv*[height<=720]+ba/b[height<=720]/b",
                "merge_output_format": "mkv",
                "outtmpl": os.path.join(tmp, "video.%(ext)s"),
            }
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                ydl.download([url])
            files = [f for f in os.listdir(tmp) if f.startswith("video.") and not f.endswith((".part", ".ytdl"))]
            if not files:
                raise RuntimeError("داونلۆدکردنی ڤیدیۆ سەرکەوتوو نەبوو.")
            src_path = os.path.join(tmp, files[0])

        w, h = probe_size(src_path)
        duration = float(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src_path],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )

        vertical = opts["layout"] == "vertical" and w >= h
        if vertical:
            cw = opts["canvas_w"]
            ch = cw * 16 // 9
            ch -= ch % 2
            vh = int(round(cw * h / w / 2)) * 2
            top = (ch - vh) // 2
            bottom = top + vh
            base_vf = (
                f"scale={cw}:-2,pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
            )
        else:
            cw, ch = w - w % 2, h - h % 2
            top, bottom = 0, ch
            base_vf = f"scale={cw}:{ch},setsar=1"

        ass = build_ass(df, title_text, opts["font"], cw, ch, vertical, top, bottom, duration)
        with open(os.path.join(tmp, "subs.ass"), "w", encoding="utf-8") as f:
            f.write(ass)

        fonts_opt = ""
        if os.path.isdir(FONTS_DIR) and os.listdir(FONTS_DIR):
            shutil.copytree(FONTS_DIR, os.path.join(tmp, "fonts"))
            fonts_opt = ":fontsdir=fonts"

        out_path = os.path.join(OUT_DIR, f"{int(time.time() * 1000)}.mp4")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-progress", "pipe:1", "-nostats",
            "-i", os.path.basename(src_path),
            "-vf", f"{base_vf},subtitles=subs.ass{fonts_opt}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            out_path,
        ]
        proc = subprocess.Popen(
            cmd, cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        tail = []
        for line in proc.stdout:
            line = line.strip()
            if line.startswith(("out_time_us=", "out_time_ms=")):
                try:
                    t = int(line.split("=", 1)[1]) / 1_000_000
                    bar.progress(0.1 + 0.88 * min(t / duration, 1.0), text="خەریکی لکاندنی ژێرنووسم بەسەر ڤیدیۆدا...")
                except ValueError:
                    pass
            elif "=" not in line and line:
                tail.append(line)
                tail = tail[-8:]
        if proc.wait() != 0 or not os.path.exists(out_path):
            raise RuntimeError("ffmpeg سەرکەوتوو نەبوو:\n" + "\n".join(tail))

    bar.progress(1.0, text="تەواو بوو")
    return out_path


# ───────────────────────── ڕووکار ─────────────────────────

DEFAULTS = {
    "layout": "vertical",
    "canvas_w": 720,
    "font": "Noto Sans Arabic",
    "max_minutes": 120,
    "whisper_model": "small",
}


def secret(name: str) -> str:
    try:
        v = st.secrets[name]
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get(name, "")


def cookies_from_secret() -> str:
    text = secret("YT_COOKIES")
    if not text:
        return ""
    ck = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    ck.write(text)
    ck.close()
    return ck.name


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if any(k in low for k in ("sign in", "confirm you", "429", "bot", "forbidden", "403")):
        return "یوتوب ڕێگری لە داونلۆدکردنی ئەم ڤیدیۆیە کرد. دووبارە هەوڵبدەرەوە، یان ڤیدیۆیەکی تر تاقی بکەرەوە."
    if "private" in low or "unavailable" in low or "not available" in low:
        return "ئەم ڤیدیۆیە تایبەتە یان بەردەست نییە."
    return msg


if "result" not in st.session_state:
    st.session_state.result = None

st.markdown(
    """
<div class="hero">
  <h1>ژێرنووسی کوردی بۆ هەر ڤیدیۆیەکی یوتوب</h1>
  <p>تەنها لینکەکە پەیست بکە. ماڵپەڕەکە خۆی قسەکانی ڤیدیۆکە دەخوێنێتەوە و ژێرنووسی کوردی سۆرانی دروست دەکات.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_up, tab_yt = st.tabs(["ئەپلۆدی ڤیدیۆ (بێ یوتوب)", "لینکی یوتوب"])

with tab_up:
    with st.form("upload_form"):
        up_file = st.file_uploader(
            "ڤیدیۆکە لێرە ئەپلۆد بکە",
            type=["mp4", "mov", "mkv", "webm", "avi", "m4v"],
        )
        up_lang_label = st.selectbox("زمانی قسەکانی ڤیدیۆ", list(LANGS.keys()), index=len(LANGS) - 1)
        up_speed = st.selectbox(
            "خێرایی",
            ["خێرا (کەمێک کەمتر وردە)", "وردتر (خاوتر)"],
            index=0,
        )
        go_up = st.form_submit_button("ژێرنووسی کوردی دروست بکە", type="primary", use_container_width=True)

with tab_yt:
    with st.form("main"):
        url = st.text_input(
            "لینکی ڤیدیۆ",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
        go = st.form_submit_button("ژێرنووسی کوردی دروست بکە", type="primary", use_container_width=True)

if go_up:
    if up_file is None:
        st.error("سەرەتا ڤیدیۆیەک ئەپلۆد بکە.")
    else:
        old = st.session_state.result
        if old and old.get("video_path") and os.path.exists(old["video_path"]):
            os.remove(old["video_path"])
        st.session_state.result = None
        api_key = get_api_key("") or secret("ANTHROPIC_API_KEY")
        uopts = {
            "src_code": LANGS[up_lang_label],
            "whisper_model": "base" if up_speed.startswith("خێرا") else DEFAULTS["whisper_model"],
            "beam_size": 1 if up_speed.startswith("خێرا") else 5,
            "engine": "claude" if api_key else "google",
            "api_key": api_key,
            "max_minutes": DEFAULTS["max_minutes"],
            "glossary": "",
        }
        with st.status("خەریکی دروستکردنی ژێرنووسم...", expanded=True) as status:
            try:
                bar = st.progress(0.0, text="دەستپێکردن...")
                st.session_state.result = process_upload(up_file, uopts, status, bar)
                status.update(label="ژێرنووسەکە ئامادەیە", state="complete", expanded=False)
            except Exception as e:
                status.update(label="سەرکەوتوو نەبوو", state="error", expanded=True)
                st.error(friendly_error(e))

if go:
    url = url.strip()
    if not url:
        st.error("لینکی ڤیدیۆکە بنووسە.")
    elif not YOUTUBE_RE.match(url):
        st.error("ئەمە لینکی یوتوب نییە. لینکەکە بە https://youtube.com یان https://youtu.be دەست پێدەکات.")
    else:
        old = st.session_state.result
        if old and old.get("video_path") and os.path.exists(old["video_path"]):
            os.remove(old["video_path"])
        st.session_state.result = None

        api_key = get_api_key("") or secret("ANTHROPIC_API_KEY")
        cookies_path = cookies_from_secret()
        opts = {
            "src_code": None,                      # زمان خۆکار دیاری دەکرێت
            "mode": "captions",                    # یەکەم ژێرنووسی یوتوب (خێرا)
            "fallback_whisper": whisper_available(),  # ئەگەر نەبوو، دەنگەکە دەخوێنێتەوە
            "whisper_model": DEFAULTS["whisper_model"],
            "engine": "claude" if api_key else "google",
            "api_key": api_key,
            "max_minutes": DEFAULTS["max_minutes"],
            "glossary": "",
            "cookies_path": cookies_path,
        }
        with st.status("خەریکی دروستکردنی ژێرنووسم...", expanded=True) as status:
            try:
                bar = st.progress(0.0, text="دەستپێکردن...")
                st.session_state.result = process_video(url, opts, status, bar)
                status.update(label="ژێرنووسەکە ئامادەیە", state="complete", expanded=False)
            except Exception as e:
                status.update(label="سەرکەوتوو نەبوو", state="error", expanded=True)
                st.error(friendly_error(e))
            finally:
                if cookies_path and os.path.exists(cookies_path):
                    os.remove(cookies_path)


res = st.session_state.result
if res:
    st.subheader(res["title"])
    dl_box = st.container()

    with st.expander("دەستکاریکردنی دەق و سەرناو"):
        st.caption("ئەگەر وشەیەک هەڵە بوو، لێرە ڕاستی بکەرەوە. فایلەکان بە دەستکارییەکەتەوە دروست دەبن.")
        edited = st.data_editor(
            res["df"],
            key=f"ed_{res['id']}_{res['rev']}",
            disabled=["start", "end", "original"],
            hide_index=True,
            num_rows="fixed",
            column_config={
                "start": st.column_config.NumberColumn("دەستپێک", format="%.1f", width="small"),
                "end": st.column_config.NumberColumn("کۆتایی", format="%.1f", width="small"),
                "original": st.column_config.TextColumn("دەقی ڕەسەن", width="medium"),
                "kurdish": st.column_config.TextColumn("کوردی", width="medium"),
            },
        )
        title_text = st.text_input(
            "سەرناوی سەرەوەی ڤیدیۆ (بە زەرد)",
            value=res["ku_title"],
            key=f"title_{res['id']}_{res['rev']}",
        )

    base = safe_name(res["title"])
    with dl_box:
        st.download_button(
            "داگرتنی ژێرنووسی کوردی (SRT)",
            build_srt(edited, "kurdish").encode("utf-8"),
            file_name=f"{base}.ckb.srt",
            mime="application/x-subrip",
            key=f"dl_ku_{res['id']}_{res['rev']}",
            type="primary",
            use_container_width=True,
        )
        st.caption("YouTube Studio ← Subtitles ← Add language ← Kurdish (Sorani) ← Upload file ← With timing")

        uploaded_video = None
        if not res.get("local_path"):
            uploaded_video = st.file_uploader(
                "ئەگەر یوتوب ڕێگری کرد: ڤیدیۆکە خۆت ئەپلۆد بکە (ئارەزوومەندانە)",
                type=["mp4", "mov", "mkv", "webm", "avi"],
                key=f"up_{res['id']}_{res['rev']}",
            )
        if st.button("دروستکردنی ڤیدیۆ بە ژێرنووسەوە", key=f"render_{res['id']}_{res['rev']}", use_container_width=True):
            cookies_path = cookies_from_secret()
            ropts = {
                "layout": DEFAULTS["layout"],
                "canvas_w": DEFAULTS["canvas_w"],
                "font": DEFAULTS["font"],
                "cookies_path": cookies_path,
            }
            rbar = st.progress(0.0, text="دەستپێکردن...")
            try:
                oldv = res.get("video_path")
                if oldv and os.path.exists(oldv):
                    os.remove(oldv)
                res["video_path"] = render_video(
                    res["url"], edited, title_text, ropts, rbar,
                    upload=uploaded_video, local_path=res.get("local_path"),
                )
            except Exception as e:
                st.error(friendly_error(e))
            finally:
                if cookies_path and os.path.exists(cookies_path):
                    os.remove(cookies_path)

        vp = res.get("video_path")
        if vp and os.path.exists(vp):
            size_mb = os.path.getsize(vp) / 1_000_000
            if size_mb < 80:
                st.video(vp)
            st.download_button(
                f"داگرتنی ڤیدیۆ (MP4، {size_mb:.0f} MB)",
                open(vp, "rb"),
                file_name=f"{base}.mp4",
                mime="video/mp4",
                key=f"dl_vid_{res['id']}_{res['rev']}",
                use_container_width=True,
            )

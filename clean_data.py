"""Clean and standardize election tweet datasets.

Cleaning criteria:
- Drop unnamed/empty columns.
- Standardize column names to: tweet_id, text, label, year.
- Set label/year based on source file.
- Normalize text by unescaping HTML, removing RT prefix, URLs, and @mentions.
- Keep hashtags but remove the leading # symbol.
- Preserve emojis and other Unicode in text.
- Normalize whitespace and lowercase.
- Filter to English-only tweets via langdetect.
- Drop empty texts and de-duplicate by tweet_id.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
from langdetect import LangDetectException, detect

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
OUTPUT_DIR = DATA_DIR / "cleaned"

FILES = [
    {
        "path": DATA_DIR / "Democrat_2020.csv",
        "label": "democrat",
        "year": 2020,
        "id_col": "tweet_id",
        "text_col": "tweet",
        "date_col": "created_at",
    },
    {
        "path": DATA_DIR / "Republican_2020.csv",
        "label": "republican",
        "year": 2020,
        "id_col": "tweet_id",
        "text_col": "tweet",
        "date_col": "created_at",
    },
    {
        "path": DATA_DIR / "Democrat_2024.csv",
        "label": "democrat",
        "year": 2024,
        "id_col": "Tweet ID",
        "text_col": "Text",
        "date_col": "Date",
    },
    {
        "path": DATA_DIR / "Republican_2024.csv",
        "label": "republican",
        "year": 2024,
        "id_col": "Tweet ID",
        "text_col": "Text",
        "date_col": "Date",
    },
]

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+", re.UNICODE)
HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)
RT_RE = re.compile(r"^\s*rt\s+", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")

EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002500-\U00002BEF"  # chinese char
    "\U00002702-\U000027B0"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F926-\U0001F937"
    "\U00010000-\U0010FFFF"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"  # dingbats
    "\u3030"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = RT_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = MULTISPACE_RE.sub(" ", text)
    text = text.strip().lower()
    return text


def is_english(text: str) -> bool:
    if not text:
        return False
    if detect is None:
        return text.isascii()
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def load_and_clean(config: dict) -> pd.DataFrame:
    df = pd.read_csv(config["path"], dtype=str)

    drop_cols = [col for col in df.columns if col.startswith("Unnamed") or col.strip() == ""]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    rename_map = {
        config["id_col"]: "tweet_id",
        config["date_col"]: "created_at",
        config["text_col"]: "text",
    }
    df = df.rename(columns=rename_map)

    df["label"] = config["label"]
    df["year"] = str(config["year"])

    df["text"] = df["text"].apply(clean_text)

    df = df[df["text"].apply(is_english)]

    df = df[df["text"].astype(bool)]

    if "tweet_id" in df.columns:
        df = df.drop_duplicates(subset=["tweet_id"], keep="first")
    else:
        df = df.drop_duplicates(subset=["text"], keep="first")

    columns = ["tweet_id", "text", "label", "year"]
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    return df[columns]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = []
    for config in FILES:
        df = load_and_clean(config)
        cleaned.append(df)

        out_name = f"{config['label']}_{config['year']}_cleaned.csv"
        df.to_csv(OUTPUT_DIR / out_name, index=False)

    combined = pd.concat(cleaned, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "tweets_all_cleaned.csv", index=False)

    combined_2020 = combined[combined["year"] == "2020"]
    combined_2024 = combined[combined["year"] == "2024"]

    combined_2020.to_csv(OUTPUT_DIR / "tweets_2020_cleaned.csv", index=False)
    combined_2024.to_csv(OUTPUT_DIR / "tweets_2024_cleaned.csv", index=False)


if __name__ == "__main__":
    main()

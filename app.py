from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import streamlit as st

try:
    from instagram_scraper import DEFAULT_NEWNESS_KEYWORDS, DEFAULT_REQUIRED_KEYWORDS
except Exception:
    DEFAULT_NEWNESS_KEYWORDS = (
        "grand opening",
        "grand re-opening",
        "soft opening",
        "soft launch",
        "now open",
        "now serving halal",
        "just opened",
        "opened today",
        "opening weekend",
        "new location",
        "new halal spot",
        "new halal restaurant",
        "new halal menu",
        "latest halal",
        "brand new",
        "halal opening",
        "halal launch",
        "coming soon",
        "opening soon",
        "open now",
        "new arrivals",
    )
    DEFAULT_REQUIRED_KEYWORDS = ("halal",)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "halal_openings.json"
SCRAPER_PATH = BASE_DIR / "instagram_scraper.py"

NEWNESS_KEYWORDS = tuple(DEFAULT_NEWNESS_KEYWORDS)
NEWNESS_SET = {kw.lower() for kw in NEWNESS_KEYWORDS}

LOCAL_TOKENS = (
    "new york",
    "nyc",
    "manhattan",
    "brooklyn",
    "queens",
    "bronx",
    "staten island",
    "long island",
    "nassau",
    "suffolk",
)
NONLOCAL_TOKENS = (
    "new jersey",
    "nj",
    "jersey city",
    "philadelphia",
    "philly",
    "pa",
    "connecticut",
    "ct",
    "boston",
    "ma",
    "maryland",
    "md",
    "virginia",
    "va",
)

ADDRESS_REPLACEMENTS = {
    "street": "st",
    "st.": "st",
    "avenue": "ave",
    "ave.": "ave",
    "road": "rd",
    "rd.": "rd",
    "boulevard": "blvd",
    "blvd.": "blvd",
    "lane": "ln",
    "ln.": "ln",
    "drive": "dr",
    "dr.": "dr",
    "court": "ct",
    "ct.": "ct",
    "place": "pl",
    "pl.": "pl",
    "parkway": "pkwy",
    "pkwy.": "pkwy",
    "highway": "hwy",
    "hwy.": "hwy",
}


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s&'-]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.startswith("the "):
        cleaned = cleaned[4:]
    return cleaned


def normalize_address(address: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", address.lower())
    cleaned = re.sub(r"\b(?:suite|ste|unit|floor|fl)\s*\w+\b", "", cleaned)
    for word, replacement in ADDRESS_REPLACEMENTS.items():
        cleaned = re.sub(rf"\b{re.escape(word)}\b", replacement, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def get_record_datetime(record: dict) -> Optional[dt.datetime]:
    raw_datetime = record.get("datetime")
    if isinstance(raw_datetime, str) and raw_datetime:
        try:
            return dt.datetime.fromisoformat(raw_datetime)
        except ValueError:
            pass
    timestamp = record.get("timestamp")
    if isinstance(timestamp, (int, float)):
        return dt.datetime.fromtimestamp(timestamp)
    raw_date = record.get("date")
    if isinstance(raw_date, str) and raw_date:
        try:
            return dt.datetime.fromisoformat(raw_date)
        except ValueError:
            pass
    return None


def get_place_name(record: dict) -> str:
    for key in ("place", "location_name", "caption_venue"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    caption = record.get("caption", "")
    if isinstance(caption, str) and caption.strip():
        return caption.splitlines()[0].strip()
    return "Unknown"


def is_new_opening(record: dict) -> bool:
    keywords = {kw.lower() for kw in (record.get("keywords") or []) if kw}
    if keywords & NEWNESS_SET:
        return True
    caption = record.get("caption", "")
    if not isinstance(caption, str):
        return False
    caption_lower = caption.lower()
    return any(kw in caption_lower for kw in NEWNESS_SET)


def text_has_token(text: str, tokens: Iterable[str]) -> bool:
    text_lower = text.lower()
    return any(token in text_lower for token in tokens)


def is_local_record(record: dict) -> bool:
    city = record.get("location_city") or ""
    address = record.get("location_address") or ""
    location = record.get("location_name") or ""
    caption = record.get("caption") or ""

    for text in (city, address, location):
        if isinstance(text, str) and text.strip():
            if text_has_token(text, NONLOCAL_TOKENS):
                return False
            if text_has_token(text, LOCAL_TOKENS):
                return True

    combined = " ".join(
        t for t in (city, address, location, caption) if isinstance(t, str) and t
    )
    if text_has_token(combined, NONLOCAL_TOKENS):
        return False
    return text_has_token(combined, LOCAL_TOKENS)


def build_search_links(place: str, address: str, city: str) -> Tuple[Optional[str], Optional[str]]:
    from urllib.parse import quote_plus

    place = place or ""
    address = address or ""
    city = city or ""
    yelp_desc = quote_plus(place) if place else ""
    yelp_loc = quote_plus(address or city) if (address or city) else ""
    yelp_url = None
    if yelp_desc or yelp_loc:
        yelp_url = f"https://www.yelp.com/search?find_desc={yelp_desc}&find_loc={yelp_loc}"

    query = " ".join(part for part in (place, address, city) if part)
    google_url = None
    if query:
        google_url = f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"

    return yelp_url, google_url


def build_group_base(record: dict) -> dict:
    place = get_place_name(record)
    address = record.get("location_address") or ""
    city = record.get("location_city") or ""
    return {
        "display_name": place,
        "address": address,
        "city": city,
        "records": [],
        "latest_dt": get_record_datetime(record) or dt.datetime.min,
    }


def add_record_to_group(group: dict, record: dict) -> None:
    group["records"].append(record)
    record_dt = get_record_datetime(record) or dt.datetime.min
    if record_dt > group["latest_dt"]:
        group["latest_dt"] = record_dt
    if not group.get("display_name") or group["display_name"] == "Unknown":
        group["display_name"] = get_place_name(record)
    if not group.get("address") and record.get("location_address"):
        group["address"] = record.get("location_address") or ""
    if not group.get("city") and record.get("location_city"):
        group["city"] = record.get("location_city") or ""


def group_records(records: List[dict], window_days: int) -> List[dict]:
    address_groups: Dict[str, dict] = {}
    no_address_records: List[dict] = []

    for record in records:
        address = record.get("location_address") or ""
        address_norm = normalize_address(address) if address else ""
        if address_norm:
            group = address_groups.get(address_norm)
            if not group:
                group = build_group_base(record)
                address_groups[address_norm] = group
            add_record_to_group(group, record)
        else:
            no_address_records.append(record)

    name_groups: Dict[str, List[dict]] = {}
    for record in sorted(no_address_records, key=lambda r: get_record_datetime(r) or dt.datetime.min, reverse=True):
        place = normalize_text(get_place_name(record))
        city = normalize_text(record.get("location_city") or "")
        key = f"{place}|{city}" if city else place
        if not key:
            key = f"unknown|{record.get('post_url') or record.get('timestamp')}"
        candidate_groups = name_groups.setdefault(key, [])

        record_dt = get_record_datetime(record) or dt.datetime.min
        assigned = None
        for group in candidate_groups:
            latest_dt = group.get("latest_dt") or dt.datetime.min
            if abs((latest_dt - record_dt).days) <= window_days:
                assigned = group
                break
        if not assigned:
            assigned = build_group_base(record)
            candidate_groups.append(assigned)
        add_record_to_group(assigned, record)

    grouped = list(address_groups.values())
    for groups in name_groups.values():
        grouped.extend(groups)
    return grouped


def summarize_group(group: dict) -> dict:
    records = group["records"]
    keywords = sorted({kw for rec in records for kw in (rec.get("keywords") or []) if kw})
    accounts = sorted({rec.get("account") for rec in records if rec.get("account")})
    newest_dt = max((get_record_datetime(rec) or dt.datetime.min for rec in records), default=dt.datetime.min)
    newest_date = newest_dt.date().isoformat() if newest_dt != dt.datetime.min else ""
    is_new = any(is_new_opening(rec) for rec in records)
    is_local = any(is_local_record(rec) for rec in records)

    display_name = group.get("display_name") or "Unknown"
    address = group.get("address") or ""
    city = group.get("city") or ""
    yelp_url, google_url = build_search_links(display_name, address, city)

    return {
        "display_name": display_name,
        "address": address,
        "city": city,
        "records": sorted(records, key=lambda r: get_record_datetime(r) or dt.datetime.min, reverse=True),
        "latest_date": newest_date,
        "keywords": keywords,
        "accounts": accounts,
        "is_new": is_new,
        "is_local": is_local,
        "yelp_url": yelp_url,
        "google_url": google_url,
    }


@st.cache_data(show_spinner=False)
def load_records(path_str: str) -> List[dict]:
    path = Path(path_str)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


@st.cache_data(show_spinner=False)
def build_groups(records: List[dict], window_days: int) -> List[dict]:
    grouped = group_records(records, window_days)
    summarized = [summarize_group(group) for group in grouped]
    summarized.sort(key=lambda g: g.get("latest_date") or "", reverse=True)
    return summarized


def run_scraper(limit: int, require_keywords: bool) -> Tuple[int, str]:
    if not SCRAPER_PATH.exists():
        return 1, f"Missing scraper at {SCRAPER_PATH}"

    cmd = [sys.executable, str(SCRAPER_PATH)]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if require_keywords:
        cmd.append("--require-keywords")
    result = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    return result.returncode, output.strip()


def render_group(group: dict, expanded: bool = False) -> None:
    title_parts = [group["display_name"]]
    if group["latest_date"]:
        title_parts.append(group["latest_date"])
    if group["is_new"]:
        title_parts.append("new opening")
    if group["is_local"]:
        title_parts.append("local")

    title = " · ".join(title_parts)
    with st.expander(title, expanded=expanded):
        cols = st.columns([3, 2])
        with cols[0]:
            if group["address"] or group["city"]:
                st.write(f"{group['address']} {group['city']}".strip())
            if group["accounts"]:
                st.caption("Accounts: " + ", ".join(group["accounts"]))
            if group["keywords"]:
                st.caption("Keywords: " + ", ".join(group["keywords"]))
        with cols[1]:
            if group["yelp_url"]:
                st.link_button("Yelp search", group["yelp_url"])
            if group["google_url"]:
                st.link_button("Google Maps search", group["google_url"])

        st.markdown("Recent posts:")
        for record in group["records"]:
            date = record.get("date") or ""
            account = record.get("account") or "unknown"
            post_url = record.get("post_url") or ""
            caption = record.get("caption") or ""
            caption = re.sub(r"\s+", " ", caption).strip()
            caption = (caption[:240] + "...") if len(caption) > 240 else caption
            st.markdown(f"- {date} · @{account} · [post]({post_url})")
    if caption:
        st.caption(caption)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Sora:wght@400;600;700&display=swap');

        :root {
          --ink: #0a2540;
          --muted: #425466;
          --surface: #ffffff;
          --surface-2: #f6f9fc;
          --border: #e6edf5;
          --brand: #635bff;
          --brand-2: #00d4ff;
          --shadow: 0 10px 30px rgba(10, 37, 64, 0.08);
          --radius: 16px;
        }

        .stApp {
          background:
            radial-gradient(1200px 700px at 10% -10%, rgba(99, 91, 255, 0.12), transparent 60%),
            radial-gradient(900px 500px at 90% 0%, rgba(0, 212, 255, 0.10), transparent 55%),
            #f6f9fc;
          color: var(--ink);
          font-family: "Instrument Sans", "Sora", "Helvetica Neue", sans-serif;
        }

        .block-container {
          max-width: 1120px;
          padding-top: 1.75rem;
        }

        h1, h2, h3 {
          font-family: "Sora", "Instrument Sans", "Helvetica Neue", sans-serif;
          letter-spacing: -0.02em;
          color: var(--ink);
        }

        p, li, span, label, div {
          color: var(--muted);
        }

        .hero {
          background: linear-gradient(120deg, rgba(99, 91, 255, 0.12), rgba(0, 212, 255, 0.08));
          border: 1px solid rgba(99, 91, 255, 0.18);
          padding: 28px 32px;
          border-radius: var(--radius);
          box-shadow: var(--shadow);
          margin-bottom: 1.6rem;
        }

        .hero h1 {
          margin: 0.4rem 0 0.6rem;
          font-size: 2.4rem;
        }

        .hero p {
          font-size: 1rem;
          margin: 0;
          max-width: 620px;
        }

        .hero-tag {
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          font-weight: 600;
          color: var(--ink);
          background: rgba(255, 255, 255, 0.7);
          border: 1px solid rgba(99, 91, 255, 0.2);
          padding: 6px 10px;
          border-radius: 999px;
        }

        .section-title {
          display: flex;
          align-items: center;
          gap: 0.6rem;
          font-size: 1.2rem;
          font-weight: 600;
          margin: 2rem 0 1rem;
          color: var(--ink);
          font-family: "Sora", "Instrument Sans", "Helvetica Neue", sans-serif;
        }

        .section-count {
          background: rgba(99, 91, 255, 0.12);
          color: var(--ink);
          border-radius: 999px;
          padding: 4px 10px;
          font-size: 0.75rem;
          font-weight: 600;
        }

        div[data-testid="metric-container"] {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 16px 18px;
          box-shadow: var(--shadow);
        }

        div[data-testid="metric-container"] label {
          color: var(--muted) !important;
          font-size: 0.8rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }

        div[data-testid="metric-container"] div {
          color: var(--ink) !important;
        }

        .stButton > button {
          background: var(--ink);
          color: white;
          border: none;
          border-radius: 999px;
          padding: 0.55rem 1.25rem;
          font-weight: 600;
          transition: transform 120ms ease, box-shadow 120ms ease;
          box-shadow: 0 12px 20px rgba(10, 37, 64, 0.12);
        }

        .stButton > button:hover {
          transform: translateY(-1px);
          box-shadow: 0 16px 26px rgba(10, 37, 64, 0.18);
        }

        .stLinkButton > a {
          border-radius: 999px;
          border: 1px solid rgba(99, 91, 255, 0.25);
          color: var(--ink);
          background: rgba(99, 91, 255, 0.08);
          font-weight: 600;
        }

        .stTextInput input, .stNumberInput input, .stTextArea textarea {
          border-radius: 12px;
          border: 1px solid var(--border);
          background: white;
        }

        .stExpander {
          border-radius: var(--radius);
          border: 1px solid var(--border);
          background: var(--surface);
          box-shadow: var(--shadow);
          margin-bottom: 0.9rem;
          overflow: hidden;
        }

        .stExpander summary {
          font-weight: 600;
          color: var(--ink);
        }

        .stExpander details[open] summary {
          border-bottom: 1px solid var(--border);
        }

        .stExpander details > summary {
          padding: 0.85rem 1rem;
        }

        .stExpander details > div {
          padding: 0.9rem 1.1rem 1.1rem;
        }

        .sidebar-content {
          background: rgba(255, 255, 255, 0.8);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Halal Openings Dashboard", layout="wide")
    inject_css()
    st.markdown(
        """
        <div class="hero">
          <div class="hero-tag">Halal openings signal</div>
          <h1>Halal Openings Dashboard</h1>
          <p>Monitor new halal restaurant openings, highlight the real signals, and keep the noise in context.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Controls")
        data_path = st.text_input("Data file", str(DATA_PATH))
        group_window = st.slider("Group window (days)", min_value=3, max_value=30, value=10)
        show_other = st.checkbox("Include non-new posts", value=True)
        show_nonlocal = st.checkbox("Include non-local locations", value=True)
        search = st.text_input("Search by name or address", "")
        st.divider()
        st.subheader("Scraper")
        limit = st.number_input("Posts per account", min_value=5, max_value=50, value=20, step=1)
        require_keywords = st.checkbox("Only save posts with new-opening keywords", value=False)
        run_now = st.button("Run scraper now")

    if run_now:
        with st.spinner("Running scraper..."):
            code, output = run_scraper(limit=limit, require_keywords=require_keywords)
        if code == 0:
            st.success("Scraper finished.")
        else:
            st.error("Scraper failed.")
        if output:
            st.code(output)
        st.cache_data.clear()

    records = load_records(data_path)
    if not records:
        st.warning("No records found. Run the scraper first.")
        return

    groups = build_groups(records, group_window)

    if search:
        query = search.lower()
        groups = [
            group
            for group in groups
            if query in group["display_name"].lower()
            or query in (group["address"] or "").lower()
            or query in (group["city"] or "").lower()
        ]

    local_new = [g for g in groups if g["is_new"] and g["is_local"]]
    local_other = [g for g in groups if (not g["is_new"]) and g["is_local"]]
    nonlocal_new = [g for g in groups if g["is_new"] and (not g["is_local"])]
    nonlocal_other = [g for g in groups if (not g["is_new"]) and (not g["is_local"])]

    total_posts = len(records)
    total_groups = len(groups)
    total_new = len([g for g in groups if g["is_new"]])
    total_local = len([g for g in groups if g["is_local"]])

    cols = st.columns(4)
    cols[0].metric("Total posts", total_posts)
    cols[1].metric("Unique places", total_groups)
    cols[2].metric("New openings", total_new)
    cols[3].metric("Local places", total_local)

    def render_section(title: str, groups_list: List[dict]) -> None:
        if not groups_list:
            st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
            st.write("No entries yet.")
            return
        st.markdown(
            f"<div class='section-title'>{title} <span class='section-count'>{len(groups_list)}</span></div>",
            unsafe_allow_html=True,
        )
        for idx, group in enumerate(groups_list):
            render_group(group, expanded=idx < 2)

    render_section("Local new openings", local_new)
    if show_other:
        render_section("Local other posts", local_other)

    if show_nonlocal:
        render_section("Other locations new openings", nonlocal_new)
        if show_other:
            render_section("Other locations other posts", nonlocal_other)


if __name__ == "__main__":
    main()

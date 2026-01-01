#!/usr/bin/env python3
"""
Instagram scraper for summarizing halal-focused posts.

Given one or more public Instagram accounts, the script fetches their
most recent posts through Instagram's web profile API, records the post
date, inferred restaurant, keywords mentioned, and optionally enforces the
halal + new-opening keyword filter before appending entries to a text
report ordered from newest to oldest post seen.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
from collections import defaultdict
from pathlib import Path
import re
import warnings
from typing import Dict, Iterable, List, Optional, Sequence
from collections import defaultdict

warnings.filterwarnings(
    "ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+", category=Warning
)

import requests


INSTAGRAM_WEB_PROFILE_URL = (
    "https://www.instagram.com/api/v1/users/web_profile_info/?username={username}"
)

# Default accounts file to read from when no accounts are supplied via CLI
DEFAULT_ACCOUNTS_FILE = Path("instagram_accounts.txt")

# Keywords: require "halal" plus at least one of the additional phrases.
DEFAULT_REQUIRED_KEYWORDS = ("halal",)
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

NON_RESTAURANT_HANDLES = {
    "_rehansyed",
    "_nursenez_",
    "aaxotics",
    "adamsaleh",
    "adamthedunker",
    "albaydargroup",
    "centercitymosque",
    "champagneadan",
    "dmv3ats",
    "dropletsofmercyusa",
    "dropletsofmercy",
    "faiz.yy",
    "faizalfilli",
    "delicatehijabi",
    "halalfoodfestdmv",
    "islamicsocietyofbaltimore",
    "jerseyhalalspots",
    "majlisofny",
    "mdq_academy",
    "mdqyouth",
    "mymasjidal",
    "muslimlightsfest",
    "muscareinc",
    "mrs.fields__tcby",
    "movingimagesusinc",
    "omarthecarguy",
    "nychalalfest",
    "razadastgir",
    "y4.m33n",
    "thecarguysny",
    "phillyhalalfoodfest",
    "phillyhalalspots",
}


def fetch_recent_posts(
    username: str,
    limit: int,
    session_headers: Optional[Dict[str, str]] = None,
    session_cookies: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Return up to `limit` recent post nodes for a public username.

    Notes:
    - Instagram's web profile API returns only the first page (typically 12 posts).
      To avoid silently truncating results, we combine that page with the user
      feed endpoint which supports pagination to satisfy `limit`.
    """
    url = INSTAGRAM_WEB_PROFILE_URL.format(username=username)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "X-IG-App-ID": "936619743392459",
        "Referer": "https://www.instagram.com/",
    }

    if session_headers:
        headers.update(session_headers)

    cookies = session_cookies or {}

    resp = requests.get(url, headers=headers, timeout=15, cookies=cookies)
    if resp.status_code != requests.codes.ok:
        raise RuntimeError(
            f"Failed to fetch profile for '{username}': HTTP {resp.status_code}"
        )

    payload = resp.json()

    # Parse initial page edges and user id when available
    edges = []
    user_id = None
    try:
        user_obj = payload["data"]["user"]
        media_section = user_obj.get("edge_owner_to_timeline_media") or {}
        edges = media_section.get("edges", []) or []
        user_id = user_obj.get("id")
    except (KeyError, TypeError):
        pass

    nodes: List[dict] = []
    seen_shortcodes: set[str] = set()
    for edge in edges:
        try:
            node = edge["node"]
        except Exception:
            continue
        sc = (node.get("shortcode") or "").strip()
        if sc and sc not in seen_shortcodes:
            nodes.append(node)
            seen_shortcodes.add(sc)
        if len(nodes) >= limit:
            return nodes[:limit]

    # If we still need more posts and have a user_id, use the paginated feed
    if user_id:
        count = min(max(limit - len(nodes), 1), 50)
        feed_url = f"https://www.instagram.com/api/v1/feed/user/{user_id}/?count={count}"
        feed_resp = requests.get(feed_url, headers=headers, timeout=15, cookies=cookies)
        if feed_resp.status_code == requests.codes.ok:
            feed_payload = feed_resp.json()
            items = list(feed_payload.get("items", []) or [])
            next_max_id = feed_payload.get("next_max_id")
            more_available = feed_payload.get("more_available")

            while len(nodes) < limit and (items or (more_available and next_max_id)):
                for item in items:
                    converted = convert_feed_item(item)
                    sc = (converted.get("shortcode") or "").strip()
                    if sc and sc not in seen_shortcodes:
                        nodes.append(converted)
                        seen_shortcodes.add(sc)
                        if len(nodes) >= limit:
                            break
                if len(nodes) >= limit or not (more_available and next_max_id):
                    break
                remaining = min(max(limit - len(nodes), 1), 50)
                paged_url = (
                    f"https://www.instagram.com/api/v1/feed/user/{user_id}/"
                    f"?count={remaining}&max_id={next_max_id}"
                )
                paged_resp = requests.get(
                    paged_url, headers=headers, timeout=15, cookies=cookies
                )
                if paged_resp.status_code != requests.codes.ok:
                    break
                paged_json = paged_resp.json()
                items = list(paged_json.get("items", []) or [])
                next_max_id = paged_json.get("next_max_id")
                more_available = paged_json.get("more_available")

    return nodes[:limit]


def convert_feed_item(item: dict) -> dict:
    """Normalize feed/user items to the graph-style node structure."""
    caption_obj = item.get("caption") or {}
    caption_text = ""
    if isinstance(caption_obj, dict):
        caption_text = caption_obj.get("text") or ""

    tag_edges = []
    usertags = item.get("usertags", {})
    if isinstance(usertags, dict):
        for tag in usertags.get("in", []) or []:
            username = (
                tag.get("user", {})
                if isinstance(tag, dict)
                else {}
            )
            if isinstance(username, dict):
                uname = username.get("username")
            else:
                uname = None
            if uname:
                tag_edges.append({"node": {"user": {"username": uname}}})

    location = item.get("location") or {}

    node = {
        "shortcode": item.get("code") or "",
        "taken_at_timestamp": item.get("taken_at"),
        "edge_media_to_caption": (
            {"edges": [{"node": {"text": caption_text}}]}
            if caption_text
            else {"edges": []}
        ),
        "edge_media_to_tagged_user": {"edges": tag_edges},
        "location": location,
    }
    return node


def extract_caption(node: dict) -> str:
    """Pull the caption text from a post node."""
    edges = node.get("edge_media_to_caption", {}).get("edges", [])
    if not edges:
        return ""
    return edges[0]["node"].get("text", "") or ""


def pick_place_name(node: dict, account_username: str, caption: str) -> Optional[str]:
    """Try to infer the place name from location metadata, tags, or caption."""
    location = node.get("location") or {}
    location_name = location.get("name")
    if isinstance(location_name, str) and location_name.strip():
        return location_name.strip()

    tagged_users = node.get("edge_media_to_tagged_user", {}).get("edges", [])
    for tag in tagged_users:
        username = (
            tag.get("node", {})
            .get("user", {})
            .get("username", "")
            .strip()
        )
        if username and username.lower() != account_username.lower():
            return f"@{username}"

    caption_line = next((line.strip() for line in caption.splitlines() if line.strip()), "")
    return caption_line or None


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Check if `keyword` exists in `text`, respecting word boundaries for single words."""
    keyword_lc = keyword.lower()
    text_lc = text.lower()
    if " " in keyword_lc:
        return keyword_lc in text_lc
    pattern = rf"\\b{re.escape(keyword_lc)}\\b"
    return re.search(pattern, text_lc) is not None


FULL_ADDRESS_REGEX = re.compile(
    r"\d{1,5}[-\w\s'&\.]*?(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Parkway|Pkwy|Highway|Hwy|Terrace|Ter|Trail|Trl|Circle|Cir|Center|Ctr)"
    r"(?:[-\w\s'&\.]*)?,\s*[A-Za-z .'-]+,\s*[A-Z]{2}(?:\s*\d{5})?",
    re.IGNORECASE,
)
STREET_FALLBACK_REGEX = re.compile(
    r"\d{1,5}[-\w\s'&\.]*(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Parkway|Pkwy|Highway|Hwy|Terrace|Ter|Trail|Trl|Circle|Cir|Center|Ctr)\b",
    re.IGNORECASE,
)
CITY_STATE_REGEX = re.compile(
    r"\b(?:New York|NYC|Brooklyn|Queens|Bronx|Staten Island|Manhattan|Long Island|Jersey City|New Jersey|NJ|Philadelphia|PA|Connecticut|CT|Boston|MA|Maryland|MD|Virginia|VA)\b",
    re.IGNORECASE,
)
VENUE_PREFIX_CLEAN = re.compile(
    r"^(?:located|location|address|find us(?: at)?|pull up to|come through to|come thru to|come to|stop by|find them at|meet us at|visit us(?: at)?|come through)\s*[:\-]*\s*",
    re.IGNORECASE,
)
VENUE_SKIP_WORDS = re.compile(
    r"\b(?:grand opening|soft opening|today|tonight|from|until|till|oct|nov|dec|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|pm|am)\b",
    re.IGNORECASE,
)
TIME_TEXT_PATTERN = re.compile(
    r"\b(?:[0-9]{1,2}\s*(?:am|pm)|[0-9]{1,2}[:.][0-9]{2}\s*(?:am|pm)|today|tonight|from\s+[0-9]|until\s+[0-9]|till\s+[0-9])\b",
    re.IGNORECASE,
)
SENTENCE_BREAK_TOKENS = [
    " #",
    " @",
    " http",
    " www",
    " Follow",
    " Give ",
    " Items",
    " MENU",
    " Menu",
    " DM ",
    " Enjoy",
    " Call",
    " Text",
    " Order",
    " Visit",
    " Check",
]


def infer_location_from_caption(
    caption: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Attempt to extract venue/address/city hints from caption text."""
    venue_candidate: Optional[str] = None
    address_candidate: Optional[str] = None
    city_candidate: Optional[str] = None
    address_confidence = -1  # -1 none, 0 weak, 1 fallback, 2 full

    def _maybe_set_city(address_text: str) -> None:
        nonlocal city_candidate
        if city_candidate:
            return
        parts = [part.strip() for part in address_text.split(",") if part.strip()]
        if len(parts) >= 2:
            city_candidate = ", ".join(parts[1:])

    for raw_line in caption.splitlines():
        line = raw_line.strip(" -*•")
        if not line:
            continue
        line = line.lstrip("📍").strip()
        if not line:
            continue
        line = VENUE_PREFIX_CLEAN.sub("", line)
        line_clean = re.sub(r"\s+", " ", line)

        def consider_venue(text: str) -> None:
            nonlocal venue_candidate
            candidate = re.sub(r"\(@[^)]+\)", "", text).strip(" ,:-")
            if not candidate:
                return
            if VENUE_SKIP_WORDS.search(candidate):
                return
            if not venue_candidate:
                venue_candidate = candidate

        if not any(char.isdigit() for char in line_clean):
            if "@" in line_clean:
                consider_venue(line_clean)
            continue

        address_text = None
        prefix_text = None
        confidence = 0
        match = FULL_ADDRESS_REGEX.search(line_clean)
        if match:
            address_text = match.group(0).strip(" ,")
            prefix_text = line_clean[: match.start()].strip(" ,:-")
            confidence = 2
        else:
            fallback_match = STREET_FALLBACK_REGEX.search(line_clean)
            if fallback_match:
                start = fallback_match.start()
                end = fallback_match.end()
                prefix_text = line_clean[:start].strip(" ,:-")
                substring = line_clean[start:]
                confidence = 1
            else:
                digit_index = next((i for i, ch in enumerate(line_clean) if ch.isdigit()), -1)
                if digit_index == -1:
                    continue
                prefix_text = line_clean[:digit_index].strip(" ,:-")
                substring = line_clean[digit_index:]
                confidence = 0

            substring = substring.strip()
            for token in SENTENCE_BREAK_TOKENS:
                pos = substring.find(token)
                if pos > 5:
                    substring = substring[:pos]
                    break
            if ". " in substring:
                substring = substring.split(". ", 1)[0]
            address_text = substring.strip(" ,")
            if TIME_TEXT_PATTERN.search(address_text) and confidence < 1:
                address_text = None

        if address_text and confidence > address_confidence:
            address_candidate = address_text
            address_confidence = confidence
            _maybe_set_city(address_text)

        if prefix_text:
            consider_venue(prefix_text)

        if not city_candidate:
            city_match = CITY_STATE_REGEX.search(line_clean)
            if city_match:
                city_candidate = city_match.group().strip(" ,")

    return venue_candidate, address_candidate, city_candidate


def process_posts(
    username: str,
    nodes: Sequence[dict],
    required_keywords: Iterable[str],
    newness_keywords: Iterable[str],
    require_keywords: bool,
) -> List[dict]:
    """Extract metadata for post nodes, optionally filtering by keyword rules."""
    required = tuple(k.lower() for k in required_keywords)
    newness = tuple(k.lower() for k in newness_keywords)

    processed: List[dict] = []
    for node in nodes:
        caption = extract_caption(node)
        caption_lower = caption.lower()

        if require_keywords:
            if not caption_lower:
                continue
            if not all(_keyword_in_text(keyword, caption_lower) for keyword in required):
                continue
            if not any(_keyword_in_text(keyword, caption_lower) for keyword in newness):
                continue

        matched_keywords = sorted(
            {
                kw
                for kw in (*required, *newness)
                if caption_lower and _keyword_in_text(kw, caption_lower)
            }
        )

        post_url = f"https://www.instagram.com/p/{node.get('shortcode', '').strip()}/"
        timestamp = node.get("taken_at_timestamp")
        post_datetime = (
            _dt.datetime.fromtimestamp(timestamp)
            if isinstance(timestamp, (int, float))
            else None
        )

        post_date = (
            post_datetime.date().isoformat()
            if isinstance(post_datetime, _dt.datetime)
            else None
        )
        location_obj = node.get("location") or {}
        location_name = location_obj.get("name")
        location_address = None
        location_city = location_obj.get("city")
        location_lat = location_obj.get("lat")
        location_lng = location_obj.get("lng")
        address_json = location_obj.get("address_json")
        if isinstance(address_json, str) and address_json:
            try:
                address_parsed = json.loads(address_json)
            except json.JSONDecodeError:
                address_parsed = {}
            if isinstance(address_parsed, dict):
                location_address = (
                    address_parsed.get("street_address")
                    or address_parsed.get("address_line1")
                )
                location_city = location_city or address_parsed.get("city_name")

        caption_venue, inferred_address, inferred_city = infer_location_from_caption(caption)
        if not location_address and inferred_address:
            location_address = inferred_address
        if not location_city and inferred_city:
            location_city = inferred_city

        tagged_edges = node.get("edge_media_to_tagged_user", {}).get("edges", [])
        tagged_accounts = []
        for tag in tagged_edges:
            username_tag = (
                tag.get("node", {})
                .get("user", {})
                .get("username", "")
            )
            if (
                username_tag
                and username_tag.lower() != username.lower()
                and username_tag not in tagged_accounts
            ):
                tagged_accounts.append(username_tag)

        place_name = pick_place_name(node, username, caption)
        caption_venue_clean = (caption_venue or "").strip()
        if caption_venue_clean:
            if (
                not place_name
                or place_name.startswith("@")
                or place_name.lower() == username.lower()
                or place_name.lower().startswith("new ")
                or place_name.lower().startswith("grand opening")
            ):
                place_name = caption_venue_clean

        processed.append(
            {
                "account": username,
                "post_url": post_url,
                "caption": caption.strip(),
                "place": place_name,
                "timestamp": timestamp,
                "datetime": post_datetime.isoformat() if post_datetime else None,
                "date": post_date,
                "keywords": matched_keywords,
                "location_name": location_name,
                "location_address": location_address,
                "location_city": location_city,
                "location_lat": location_lat,
                "location_lng": location_lng,
                "tagged_accounts": tagged_accounts,
                "caption_venue": caption_venue_clean or None,
            }
        )

    # Keep newest posts first based on timestamp if present.
    return sorted(processed, key=lambda item: item.get("timestamp") or 0, reverse=True)


def _clean_caption(text: str) -> str:
    """Collapse whitespace to keep captions readable in single-line outputs."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def load_existing_records(path: Path) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def merge_records(existing: Sequence[dict], new_records: Sequence[dict]) -> List[dict]:
    merged: dict[str, dict] = {}
    order: List[str] = []

    def upsert(record: dict) -> None:
        key = record.get("post_url") or f"{record.get('account')}|{record.get('timestamp')}"
        if key in merged:
            merged[key] = record
        else:
            merged[key] = record
            order.append(key)

    for item in existing:
        upsert(item)
    for item in new_records:
        upsert(item)

    ordered_records = [merged[key] for key in order]
    ordered_records.sort(key=lambda item: item.get("timestamp") or 0, reverse=True)
    return ordered_records


def write_json(records: Sequence[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


def write_csv(records: Sequence[dict], output_path: Path) -> None:
    """Write post details to CSV for spreadsheet analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "captured_date",
        "posted_date",
        "account",
        "keywords",
        "venue",
        "post_url",
        "location_address",
        "location_city",
        "tagged_accounts",
        "caption",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in records:
            keywords = "; ".join(item.get("keywords") or [])
            venue = item.get("place") or item.get("location_name") or ""
            row = {
                "captured_date": item.get("run_date") or "",
                "posted_date": item.get("date") or "",
                "account": item.get("account") or "",
                "keywords": keywords,
                "venue": venue,
                "post_url": item.get("post_url") or "",
                "location_address": item.get("location_address") or "",
                "location_city": item.get("location_city") or "",
                "tagged_accounts": "; ".join(item.get("tagged_accounts") or []),
                "caption": _clean_caption(item.get("caption", "")),
            }
            writer.writerow(row)


def write_excel(
    records: Sequence[dict],
    output_path: Path,
    monitored_accounts: Optional[Sequence[str]] = None,
) -> None:
    """Write an Excel workbook with highlighted keyword hits."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill
    except ImportError:
        print("[warn] openpyxl not installed; skipping Excel export.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Halal Openings"

    headers = [
        "captured_date",
        "posted_date",
        "account",
        "keywords",
        "venue",
        "post_url",
        "location_address",
        "location_city",
        "tagged_accounts",
        "caption",
    ]
    ws.append(headers)

    highlight = PatternFill(start_color="C7F5B4", end_color="C7F5B4", fill_type="solid")

    for item in records:
        keywords = "; ".join(item.get("keywords") or [])
        row = [
            item.get("run_date") or "",
            item.get("date") or "",
            item.get("account") or "",
            keywords,
            item.get("place") or item.get("location_name") or "",
            item.get("post_url") or "",
            item.get("location_address") or "",
            item.get("location_city") or "",
            "; ".join(item.get("tagged_accounts") or []),
            _clean_caption(item.get("caption", "")),
        ]
        ws.append(row)
        if keywords:
            for cell in ws[ws.max_row]:
                cell.fill = highlight

    accounts_to_ignore = {acc.lower() for acc in (monitored_accounts or [])} | NON_RESTAURANT_HANDLES
    accounts_sheet = wb.create_sheet("Tagged Accounts")
    accounts_sheet.append(
        ["tagged_account", "times_tagged", "latest_posted_date", "accounts_posted", "post_urls"]
    )

    tagged_map: Dict[str, Dict[str, set]] = defaultdict(
        lambda: {"posts": set(), "accounts": set(), "dates": set()}
    )
    for record in records:
        account = record.get("account") or ""
        post_url = record.get("post_url") or ""
        posted_date = record.get("date") or ""
        for tag in record.get("tagged_accounts") or []:
            if tag.lower() in accounts_to_ignore:
                continue
            key = tag.strip()
            if not key:
                continue
            tagged_map[key]["posts"].add(post_url)
            tagged_map[key]["accounts"].add(account)
            if posted_date:
                tagged_map[key]["dates"].add(posted_date)

    for tag, info in sorted(tagged_map.items()):
        latest_date = ""
        if info["dates"]:
            latest_date = max(info["dates"])
        accounts_list = "; ".join(sorted(acc for acc in info["accounts"] if acc))
        post_list = "; ".join(sorted(url for url in info["posts"] if url))
        accounts_sheet.append(
            [
                tag,
                len([url for url in info["posts"] if url]),
                latest_date,
                accounts_list,
                post_list,
            ]
        )

    wb.save(output_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize recent Instagram posts with optional halal keyword filtering."
    )
    parser.add_argument(
        "--accounts",
        nargs="+",
        help="One or more Instagram usernames to scan (e.g. halaleatsig).",
    )
    parser.add_argument(
        "--accounts-file",
        type=Path,
        help=(
            "Optional path to a text file with one username per line. "
            "If omitted, the script will use 'instagram_accounts.txt' when present."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of recent posts per account to inspect (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("halal_openings.json"),
        help="JSON file to store aggregated matches (default: halal_openings.json).",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("halal_openings.csv"),
        help="CSV file to append matches to (default: halal_openings.csv).",
    )
    parser.add_argument(
        "--xlsx-output",
        type=Path,
        default=Path("halal_openings.xlsx"),
        help="Optional Excel workbook to overwrite with the latest results.",
    )
    parser.add_argument(
        "--sessionid",
        help="Instagram sessionid cookie value (or set IG_SESSIONID env var).",
    )
    parser.add_argument(
        "--csrftoken",
        help="Instagram csrftoken cookie value (or set IG_CSRFTOKEN env var).",
    )
    parser.add_argument(
        "--require-keywords",
        action="store_true",
        help="Only record posts that include halal + new-opening keywords.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matches to stdout without writing the output file.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Dump match JSON to stdout for debugging.",
    )
    return parser.parse_args(argv)


def load_accounts(args: argparse.Namespace) -> List[str]:
    """Load usernames from CLI and/or a default file; normalize @prefixes."""
    accounts: List[str] = []

    # Direct CLI usernames
    if args.accounts:
        accounts.extend(args.accounts)

    # Prefer explicit file if provided; otherwise fall back to DEFAULT_ACCOUNTS_FILE if present
    files_to_read: List[Path] = []
    if args.accounts_file:
        files_to_read.append(args.accounts_file)
    elif DEFAULT_ACCOUNTS_FILE.exists():
        files_to_read.append(DEFAULT_ACCOUNTS_FILE)

    for file_path in files_to_read:
        try:
            for line in file_path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                # Skip comments and blanks
                if not raw or raw.startswith("#"):
                    continue
                # Strip leading @ if present
                if raw.startswith("@"):  # tolerate @username format
                    raw = raw[1:]
                accounts.append(raw)
        except OSError:
            # Ignore unreadable/missing files silently; caller handles empty case
            pass

    # Normalize and dedupe
    return sorted({acc.lstrip("@").lower() for acc in accounts if acc})


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    accounts = load_accounts(args)
    if not accounts:
        print("No accounts provided. Use --accounts or --accounts-file.", flush=True)
        return 1

    sessionid = args.sessionid or os.getenv("IG_SESSIONID")
    csrftoken = args.csrftoken or os.getenv("IG_CSRFTOKEN")
    session_cookies: Dict[str, str] = {}
    session_headers: Dict[str, str] = {}

    if sessionid:
        session_cookies["sessionid"] = sessionid
    if csrftoken:
        session_cookies["csrftoken"] = csrftoken
        session_headers["X-CSRFToken"] = csrftoken

    if session_cookies:
        session_headers.setdefault("X-Requested-With", "XMLHttpRequest")

    run_date = _dt.date.today().isoformat()
    all_records: List[dict] = []
    for account in accounts:
        try:
            nodes = fetch_recent_posts(
                account,
                args.limit,
                session_headers=session_headers or None,
                session_cookies=session_cookies or None,
            )
        except Exception as exc:
            print(f"[error] {account}: {exc}", flush=True)
            continue

        records = process_posts(
            username=account,
            nodes=nodes,
            required_keywords=DEFAULT_REQUIRED_KEYWORDS,
            newness_keywords=DEFAULT_NEWNESS_KEYWORDS,
            require_keywords=args.require_keywords,
        )
        for record in records:
            record["run_date"] = run_date

        if not records:
            if args.require_keywords:
                print(f"[info] {account}: no posts matched the keyword rules.", flush=True)
            else:
                print(f"[info] {account}: no posts were collected.", flush=True)
            continue

        keyword_hits = sum(1 for item in records if item.get("keywords"))
        print(
            f"[info] {account}: recorded {len(records)} posts "
            f"(keyword hits: {keyword_hits}).",
            flush=True,
        )
        all_records.extend(records)

        if args.show_json:
            print(json.dumps(records, indent=2, ensure_ascii=False))

    if args.dry_run:
        if all_records:
            print(json.dumps(all_records, indent=2, ensure_ascii=False))
        else:
            message = (
                "No posts matched the keyword criteria."
                if args.require_keywords
                else "No posts recorded."
            )
            print(message)
        return 0

    if not all_records:
        if args.require_keywords:
            print("No posts matched the keyword criteria.")
        else:
            print("No posts recorded.")
        return 0

    existing_records = load_existing_records(args.output)
    merged_records = merge_records(existing_records, all_records)

    write_json(merged_records, args.output)
    write_csv(merged_records, args.csv_output)
    if args.xlsx_output:
        write_excel(merged_records, args.xlsx_output, monitored_accounts=accounts)

    matched_accounts = sorted({item["account"] for item in merged_records})
    new_accounts = ", ".join(sorted({item["account"] for item in all_records}) or ["none"])
    print(
        f"Merged {len(merged_records)} total entries across accounts: {', '.join(matched_accounts)} "
        f"(added {len(all_records)} new from: {new_accounts}) "
        f"into {args.output}, {args.csv_output}"
        + (f", and {args.xlsx_output}" if args.xlsx_output else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

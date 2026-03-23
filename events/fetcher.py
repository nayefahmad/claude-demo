"""Fetch events from external APIs with a mock fallback for testing."""
import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

TICKETMASTER_BASE = "https://app.ticketmaster.com/discovery/v2"
EVENTBRITE_BASE = "https://www.eventbriteapi.com/v3"


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _event_id(source: str, raw_id: str) -> str:
    return f"{source}:{raw_id}"


# ---------------------------------------------------------------------------
# Ticketmaster
# ---------------------------------------------------------------------------

def _fetch_ticketmaster(days_ahead: int = 14) -> list[dict]:
    if not config.TICKETMASTER_API_KEY:
        return []
    start = datetime.utcnow()
    end = start + timedelta(days=days_ahead)
    params = {
        "apikey": config.TICKETMASTER_API_KEY,
        "city": config.CITY,
        "countryCode": config.COUNTRY_CODE,
        "startDateTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endDateTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "size": 100,
        "sort": "date,asc",
    }
    try:
        resp = requests.get(f"{TICKETMASTER_BASE}/events.json", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Ticketmaster fetch failed: %s", exc)
        return []

    events = []
    for item in data.get("_embedded", {}).get("events", []):
        dates = item.get("dates", {}).get("start", {})
        price_ranges = item.get("priceRanges", [{}])
        venue_info = item.get("_embedded", {}).get("venues", [{}])[0]
        classifications = item.get("classifications", [{}])[0]
        segment = classifications.get("segment", {}).get("name", "")
        genre = classifications.get("genre", {}).get("name", "")
        category = f"{segment} - {genre}".strip(" -")
        images = item.get("images", [])
        image_url = images[0]["url"] if images else None
        events.append({
            "id": _event_id("tm", item["id"]),
            "source": "ticketmaster",
            "title": item.get("name", ""),
            "description": item.get("info") or item.get("pleaseNote") or "",
            "category": category,
            "start_dt": dates.get("dateTime", dates.get("localDate", "")),
            "end_dt": "",
            "venue": venue_info.get("name", ""),
            "url": item.get("url", ""),
            "image_url": image_url,
            "price_min": price_ranges[0].get("min") if price_ranges else None,
            "price_max": price_ranges[0].get("max") if price_ranges else None,
            "fetched_at": _now_iso(),
        })
    logger.info("Ticketmaster: fetched %d events", len(events))
    return events


# ---------------------------------------------------------------------------
# Eventbrite
# ---------------------------------------------------------------------------

def _fetch_eventbrite(days_ahead: int = 14) -> list[dict]:
    if not config.EVENTBRITE_TOKEN:
        return []
    start = datetime.utcnow()
    end = start + timedelta(days=days_ahead)
    params = {
        "location.address": config.CITY,
        "location.within": "25mi",
        "start_date.range_start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_date.range_end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expand": "venue,ticket_classes",
        "page_size": 50,
    }
    headers = {"Authorization": f"Bearer {config.EVENTBRITE_TOKEN}"}
    try:
        resp = requests.get(f"{EVENTBRITE_BASE}/events/search/",
                            params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Eventbrite fetch failed: %s", exc)
        return []

    events = []
    for item in data.get("events", []):
        venue = item.get("venue") or {}
        tickets = item.get("ticket_classes", [{}])
        price_min = None
        price_max = None
        for t in tickets:
            cost = t.get("cost")
            if cost:
                val = float(cost.get("major_value", 0))
                if price_min is None or val < price_min:
                    price_min = val
                if price_max is None or val > price_max:
                    price_max = val
        events.append({
            "id": _event_id("eb", item["id"]),
            "source": "eventbrite",
            "title": item.get("name", {}).get("text", ""),
            "description": (item.get("description") or {}).get("text", "")[:500],
            "category": item.get("category_id", ""),
            "start_dt": item.get("start", {}).get("utc", ""),
            "end_dt": item.get("end", {}).get("utc", ""),
            "venue": venue.get("name", ""),
            "url": item.get("url", ""),
            "image_url": (item.get("logo") or {}).get("url"),
            "price_min": price_min,
            "price_max": price_max,
            "fetched_at": _now_iso(),
        })
    logger.info("Eventbrite: fetched %d events", len(events))
    return events


# ---------------------------------------------------------------------------
# Mock data (used when no API keys are configured)
# ---------------------------------------------------------------------------

_MOCK_CATEGORIES = [
    "Music - Rock",
    "Music - Jazz",
    "Arts & Theatre - Theatre",
    "Arts & Theatre - Comedy",
    "Sports - Soccer",
    "Sports - Basketball",
    "Food & Drink - Festival",
    "Technology - Conference",
    "Family - Kids",
    "Outdoor - Hiking",
]

_MOCK_VENUES = [
    "City Arena",
    "Downtown Jazz Club",
    "Community Theatre",
    "Convention Center",
    "Waterfront Park",
    "Historic Ballroom",
    "Rooftop Lounge",
    "University Hall",
    "Night Market Square",
    "Botanical Gardens",
]

_MOCK_EVENTS_TEMPLATES = [
    ("Rock the Block: Local Bands Live", "Music - Rock",
     "Three local rock bands take the stage for a night of original music. "
     "Expect high-energy performances, light shows, and merchandise."),
    ("Jazz Under the Stars", "Music - Jazz",
     "An outdoor jazz evening featuring a six-piece ensemble playing classics "
     "and originals. Bring a blanket and enjoy cocktails under the open sky."),
    ("Improv Comedy Showcase", "Arts & Theatre - Comedy",
     "The city's best improv troupes compete in a hilarious evening of "
     "audience-driven comedy. No two shows are the same!"),
    ("Family Fun Festival", "Family - Kids",
     "A weekend celebration for all ages with carnival rides, face painting, "
     "live music, and local food vendors. Free entry for children under 12."),
    ("Tech Startup Mixer", "Technology - Conference",
     "Monthly networking event for founders, engineers, and investors. "
     "Includes lightning talks, demo tables, and open bar."),
    ("Chef's Table: Farm-to-Fork Dinner", "Food & Drink - Festival",
     "A six-course tasting menu prepared by award-winning local chefs using "
     "ingredients sourced entirely within 50 miles. Wine pairing available."),
    ("City Half Marathon", "Sports - Running",
     "Annual 13.1-mile race through scenic city streets and parks. "
     "All fitness levels welcome. Post-race refreshments and medals for finishers."),
    ("Outdoor Cinema Night", "Arts & Theatre - Film",
     "Bring your picnic blanket and watch a classic film projected on a giant "
     "outdoor screen. Food trucks on site. Doors open at sunset."),
    ("Contemporary Art Opening", "Arts & Theatre - Visual Arts",
     "Opening night for a new exhibition featuring 20 emerging local artists "
     "working across painting, sculpture, and digital media. Wine reception included."),
    ("Yoga in the Park", "Outdoor - Wellness",
     "Free community yoga session led by certified instructors. "
     "All levels welcome. Bring your own mat. Donations appreciated."),
    ("Salsa Dancing Workshop", "Arts & Theatre - Dance",
     "Learn the fundamentals of salsa in this beginner-friendly two-hour workshop. "
     "No partner required. Includes a social dance session afterward."),
    ("Craft Beer Festival", "Food & Drink - Festival",
     "Sample over 100 craft beers from 40+ local and regional breweries. "
     "Live music, food trucks, and homebrewing demonstrations throughout the day."),
    ("Book Club Meetup: Sci-Fi Month", "Arts & Theatre - Literature",
     "Monthly gathering of the city's largest book club. This month: "
     "discussing a seminal science-fiction novel. New members always welcome."),
    ("Farmers Market & Cooking Demo", "Food & Drink - Market",
     "The weekly farmers market features a live cooking demo by a local chef "
     "using seasonal produce available at the stalls."),
    ("Photography Walk Downtown", "Arts & Theatre - Photography",
     "Guided photography walk through historic downtown. "
     "Learn composition and street photography techniques from a professional photographer."),
]


def _generate_mock_events(days_ahead: int = 14) -> list[dict]:
    events = []
    base = datetime.utcnow()
    for i, (title, category, description) in enumerate(_MOCK_EVENTS_TEMPLATES):
        days_offset = (i * days_ahead) // len(_MOCK_EVENTS_TEMPLATES)
        hour = 10 + (i % 12)
        start_dt = base + timedelta(days=days_offset, hours=hour)
        raw_id = hashlib.md5(f"{title}-{start_dt.isoformat()}".encode()).hexdigest()[:12]
        events.append({
            "id": _event_id("mock", raw_id),
            "source": "mock",
            "title": title,
            "description": description,
            "category": category,
            "start_dt": start_dt.isoformat() + "Z",
            "end_dt": (start_dt + timedelta(hours=2)).isoformat() + "Z",
            "venue": _MOCK_VENUES[i % len(_MOCK_VENUES)],
            "url": f"https://example.com/events/{raw_id}",
            "image_url": None,
            "price_min": [0, 10, 15, 25, 50, 75][i % 6],
            "price_max": [0, 20, 30, 50, 100, 150][i % 6],
            "fetched_at": _now_iso(),
        })
    logger.info("Mock: generated %d events", len(events))
    return events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_events(days_ahead: int = 14) -> list[dict]:
    """Fetch events from all configured sources; fall back to mock data."""
    events: list[dict] = []
    events.extend(_fetch_ticketmaster(days_ahead))
    events.extend(_fetch_eventbrite(days_ahead))
    if not events:
        logger.info("No API keys configured – using mock event data.")
        events = _generate_mock_events(days_ahead)
    # Deduplicate by id
    seen: set[str] = set()
    unique: list[dict] = []
    for e in events:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique.append(e)
    return unique

"""
GreekRank Forum Scraper - Skeleton
===================================
A polite, beginner-friendly web scraper for collecting forum post text
from GreekRank for later word-frequency analysis.

BEFORE RUNNING:
  1. Check https://www.greekrank.com/robots.txt for any Disallow rules
     covering the paths you plan to hit.
  2. Skim the Terms of Service. If scraping is forbidden, stop here.
  3. Install dependencies:
        pip install requests beautifulsoup4 pandas tqdm

WHAT THIS FILE DOES:
  - Fetches one or more forum/thread pages from GreekRank.
  - Parses the HTML and pulls out the text of each post.
  - Saves everything to a CSV so the next script (text-analysis) can
    load it without re-scraping.

WHAT IT DOES NOT DO (yet):
  - Word counting, stopword removal, plotting. That's the next stage.
  - JavaScript rendering. If GreekRank loads posts via JS, you'll need
    Playwright or Selenium instead — we'll detect this and warn you.
"""

import csv          # writing rows to a CSV file
import time         # sleeping between requests (politeness)
import random       # jittering the sleep so we don't look like a robot metronome
from pathlib import Path
from urllib.parse import urljoin   # safely build full URLs from relative links

import requests                    # fetches web pages
from bs4 import BeautifulSoup      # parses HTML into something searchable
from tqdm import tqdm              # progress bar; nice for long runs


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Putting "magic values" up top makes them easy to tweak without hunting
# through the code. As a data scientist you'll see this pattern constantly.

BASE_URL = "https://www.greekrank.net"

# A list of starting URLs to scrape. Start SMALL — one or two pages — until
# you've confirmed the parser works. Then expand.
# Example pattern for Purdue fraternity discussion (verify in your browser
# before relying on it; GreekRank's URL structure may differ).
SEED_URLS = [
    "https://www.greekrank.net/uni/48/discussion/",
]

# How long to wait between requests, in seconds. Be generous. A real human
# browsing reads for several seconds between page loads — your scraper
# should look at least that polite.
MIN_DELAY = 3.0
MAX_DELAY = 6.0

# Where the scraped data lands.
OUTPUT_CSV = Path("greekrank_posts.csv")

# A realistic User-Agent header. Many sites refuse requests from the default
# "python-requests/2.x" UA. Using a browser UA isn't deceptive on its own,
# but combined with ignoring robots.txt it can be — so only use this on
# sites that allow scraping.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------
def fetch_page(url: str, session: requests.Session) -> str | None:
    """
    Download a single page and return its HTML as a string.

    Returns None on failure so callers can skip-and-continue rather than crash
    the whole run. In data work you'll often process hundreds of inputs and
    accept that a few will fail — designing for partial failure from the
    start saves a lot of pain.
    """
    try:
        # `timeout` matters: without it, a slow server can hang your script
        # forever. 15 seconds is a reasonable upper bound for a page load.
        response = session.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()   # raises an exception on 4xx/5xx status
        return response.text
    except requests.RequestException as e:
        # Catching the broad RequestException covers timeouts, DNS errors,
        # HTTP errors, etc. — anything `requests` itself might raise.
        print(f"  [!] Failed to fetch {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# PARSE
# ---------------------------------------------------------------------------
def parse_posts(html: str, source_url: str) -> list[dict]:
    """
    Extract individual posts from a forum/discussion page.

    Returns a list of dicts, one per post. Using dicts (rather than tuples)
    means each field is named — when you load into pandas later, those names
    become column headers automatically.

    Selectors are based on GreekRank's actual HTML (discussion-box-*).
    If the site changes its markup, you'll need to update these.
    """
    soup = BeautifulSoup(html, "html.parser")
    posts = []

    # GreekRank wraps each thread/discussion in <div class="discussion-box">
    # (often with "clearfix" added — but we match on "discussion-box" alone
    # so the selector works either way). Inside each one:
    #   <h5 class="discussion-box-head"> title text </h5>
    #   <div class="discussion-box-content"> body text </div>
    #
    # Looping over the outer wrapper (rather than collecting all titles and
    # all bodies separately) is more robust — if one post is missing a piece,
    # we just record empty values for that one instead of misaligning the
    # whole list.
    discussion_boxes = soup.select("div.discussion-box")

    if not discussion_boxes:
        # Helpful diagnostic: if zero posts are found, either the selector
        # is wrong OR the page renders posts via JavaScript (in which case
        # `requests` sees an empty shell and you need Playwright/Selenium).
        print(f"  [!] No discussion boxes found on {source_url}. "
              "Either the CSS selector is wrong or the page is JS-rendered.")
        return posts

    for box in discussion_boxes:
        # Title lives in an <h5> with class "discussion-box-head".
        head = box.select_one("h5.discussion-box-head")
        title = head.get_text(strip=True) if head else ""

        # Body lives in <div class="discussion-box-content">.
        # get_text(separator=" ") joins child elements (multiple <p>s, <span>s,
        # etc.) with spaces so the body reads as one continuous string instead
        # of "word1word2" smushed together. strip=True trims whitespace.
        content = box.select_one("div.discussion-box-content")
        body = content.get_text(separator=" ", strip=True) if content else ""

        # Skip totally empty entries (sometimes ads or section dividers can
        # match the outer selector but have no real content).
        if not title and not body:
            continue

        posts.append({
            "source_url": source_url,
            "title": title,
            "body": body,
        })

    return posts


def find_pagination_links(html: str, current_url: str) -> list[str]:
    """
    Find links to additional pages of the same discussion list.

    GreekRank's pagination lives in <div class="post-pagination">, which
    contains anchor tags for page numbers and a "NEXT >" / "LAST" link.
    We grab every <a> inside that container.

    We deduplicate the returned list because page 2 and "NEXT >" might
    point to the same URL — no point queueing it twice.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []
    seen = set()
    for a in soup.select("div.post-pagination a"):
        href = a.get("href")
        if not href:
            continue
        # urljoin handles both absolute and relative URLs correctly.
        # If href is "/uni/48/discussion/?page=2" it becomes a full URL.
        # If href is already absolute, it's returned unchanged.
        full = urljoin(current_url, href)
        if full not in seen:
            seen.add(full)
            links.append(full)
    return links


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
def save_to_csv(rows: list[dict], path: Path) -> None:
    """
    Append rows to a CSV file, writing a header row only on first write.

    Appending (rather than overwriting) means you can run the scraper in
    chunks. If it crashes halfway through, you don't lose what you already
    collected.
    """
    if not rows:
        return

    file_exists = path.exists()
    # `newline=""` prevents Python from inserting extra blank lines on Windows.
    # `encoding="utf-8"` keeps emoji / unusual characters from breaking things.
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# MAIN ORCHESTRATION
# ---------------------------------------------------------------------------
def scrape(seed_urls: list[str], output_path: Path) -> None:
    """
    Top-level driver. Loops over seed URLs, fetches each, parses posts,
    follows pagination, and saves results.
    """
    if not seed_urls:
        print("No seed URLs configured. Edit SEED_URLS at the top of this file.")
        return

    # A Session reuses the underlying TCP connection across requests, which
    # is faster and lighter on the server than opening a fresh connection
    # for every page. It also persists cookies, which some sites require.
    with requests.Session() as session:
        # Build a working queue of URLs. A set tracks what we've already
        # done so we don't loop forever on circular pagination links.
        to_visit = list(seed_urls)
        visited: set[str] = set()

        # tqdm gives a live progress bar. Helpful psychologically on long runs.
        pbar = tqdm(total=len(to_visit), desc="Scraping")

        while to_visit:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)

            html = fetch_page(url, session)
            if html is None:
                pbar.update(1)
                continue

            posts = parse_posts(html, source_url=url)
            save_to_csv(posts, output_path)

            # Queue up next pages discovered on this one.
            for next_url in find_pagination_links(html, url):
                if next_url not in visited:
                    to_visit.append(next_url)
                    pbar.total += 1
                    pbar.refresh()

            pbar.update(1)

            # POLITENESS: sleep before the next request. Random jitter avoids
            # a perfectly regular pattern (which can trip rate-limit detectors
            # and also overloads servers in predictable bursts).
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

        pbar.close()

    print(f"\nDone. Data written to {output_path.resolve()}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
# The `if __name__ == "__main__":` guard means this block only runs when
# you execute the file directly (`python greekrank_scraper.py`) — not when
# another script imports it. Good habit even for one-file projects.
if __name__ == "__main__":
    scrape(SEED_URLS, OUTPUT_CSV)

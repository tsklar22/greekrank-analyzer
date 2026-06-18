# GreekRank Forum Analyzer

An NLP pipeline that scrapes, cleans, and analyzes forum posts from
[GreekRank](https://www.greekrank.net) to surface word frequency patterns
and sentiment trends across Greek life discussions.

Built as a personal data science project combining web scraping,
text analysis, and data visualization.

## Features

- **Polite web scraper** — collects forum posts with rate-limited
  requests and automatic pagination following
- **Text cleaning pipeline** — regex-based site-chrome removal,
  custom stopword filtering, and Greek letter / chapter abbreviation
  classification
- **Word frequency analysis** — separate charts for topic words vs.
  Greek chapter names, plus bigram (two-word phrase) frequency
- **Chapter mention analyzer** — filter the corpus to posts mentioning
  a specific fraternity and analyze language patterns in context
- **Sentiment analysis** — VADER-based scoring with distribution
  histograms and cross-group comparison boxplots

## Output Examples

*(add a screenshot or two of your charts here once you have them)*

## Tech Stack

- `requests` + `BeautifulSoup` — scraping
- `nltk` — stopwords, bigrams, VADER sentiment
- `pandas` — data handling
- `matplotlib` — visualization
- `wordcloud` — word cloud generation

## Setup

```bash
pip install requests beautifulsoup4 nltk pandas matplotlib wordcloud
```

**Scrape data:**
```bash
python greekrank_scraper.py
```

**Run analysis:**
```bash
python greekrank_analyze.py
```

## Notes

This project is for personal/educational use. GreekRank's `robots.txt`
permits crawling (`Disallow:` is blank). Requests are rate-limited to
3–6 seconds between pages out of respect for the server.
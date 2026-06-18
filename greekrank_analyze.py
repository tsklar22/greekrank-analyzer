"""
GreekRank Text Analysis & Visualization (v3)
=============================================
NEW IN v3: Sentiment analysis
  - Each post gets scored on a -1 (very negative) to +1 (very positive) scale
  - Distribution histograms showing how positive/negative the corpus is
  - Comparison: are AEPi-mentioning posts more positive or negative than average?
  - The 5 most positive and 5 most negative posts per category, printed for review

WHY VADER?
  VADER (Valence Aware Dictionary and sEntiment Reasoner) is a sentiment
  analyzer built specifically for short social-media-style text. It
  understands slang, emphasis ("really good!" > "good"), and negation
  ("not great") in ways simpler tools don't. Perfect for forum posts.

BEFORE RUNNING:
  pip install pandas matplotlib wordcloud nltk
"""

import re
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.sentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_CSV = Path("greekrank_posts.csv")
OUTPUT_DIR = Path("analysis_output")

TOP_N = 25
MIN_WORD_LENGTH = 2

# --- NEW: Chapter-mention focus ---
# Set this to a chapter name (or any keyword). The script will additionally
# produce a SEPARATE analysis showing top words in ONLY the posts that
# mention this chapter. Use a list of synonyms to catch all variants of
# the name people use. Case-insensitive.
#
# To skip this feature, set FOCUS_CHAPTER = None.
FOCUS_CHAPTER = {
    "name": "AEPi",                              # label used in chart titles
    "aliases": ["aepi", "ae pi", "alpha epsilon pi", "aep"],   # all variants
}


# ---------------------------------------------------------------------------
# PRE-TOKENIZE CLEANUP PATTERNS
# ---------------------------------------------------------------------------
# These regex patterns run on the RAW text BEFORE we extract words. They
# remove site chrome that the scraper couldn't fully strip out — phrases
# that get glued to real words and create junk tokens like "pledgelast".
#
# We replace each match with a single space so the surrounding real words
# stay separated. Patterns are case-insensitive (re.IGNORECASE flag below).
SITE_CHROME_PATTERNS = [
    r"Read More",
    r"Last Post\s*:?\s*[\d\w]*\s*(year|month|week|day|hour|minute)s?\s*ago",
    r"Last Post",
    r"By\s*:\s*[A-Za-z0-9_]+",            # "By: SomeUsername"
    r"\d+\s*replies?",                     # "5 replies"
    r"\d+\s*views?",                       # "331 Views"
    r"Started\s*:\s*[A-Za-z]+\s*\d+,?\s*\d*",   # "Started: Jan 6, 2015"
    r"\d{1,2}:\d{2}:\d{2}\s*(AM|PM)?",     # timestamps like 11:53:19 PM
    r"\d{1,2}/\d{1,2}/\d{2,4}",            # dates like 1/6/2015
]


# ---------------------------------------------------------------------------
# GREEK ALPHABET + CHAPTER ABBREVIATIONS
# ---------------------------------------------------------------------------
GREEK_LETTERS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
}

FRAT_SORORITY_ABBREVIATIONS = {
    # IFC fraternities
    "sae", "sigep", "phipsi", "phidelt", "phigam", "fiji",
    "deltsig", "deltchi", "dchi", "dke", "dku", "tke", "teke",
    "atokap", "ato", "akl", "pike", "pikapp", "pikappa", "pikapps", "psiu",
    "betatp", "betatpi", "lxa", "lambdachi", "lca",
    "zbt", "ze",
    "ksig", "kappasig", "kapps", "kapp",
    "akpsi", "apo", "thetachi", "thetax", "tx",
    "sigchi", "sigmachi", "sigtau", "sigmatau", "snu", "tc", "delts", "delt",
    "aepi", "aep", "sig", "sigs",
    "acacia", "triangle", "farmhouse",
    "du",        # Delta Upsilon
    "dsig",      # Delta Sigma Phi
    "agr",       # Alpha Gamma Rho
    "tri", "agd", "asig", "dsig", "kicked", "probation",
    "probo",     # caught in earlier output
    # Panhellenic sororities
    "aoii", "aopi", "aphi", "alphaphi", "kkg",
    "tridelt", "tridelta", "ddd", "ade", "adpi", "alphadeltpi",
    "chio", "chiomega", "tritri", "kd", "kappadelta",
    "pibphi", "pibetphi", "pibetaphi", "gphib", "gammaphibeta",
    "thetakap", "kao", "kkapg", "zta", "zetatau", "zetataualpha",
    "dg", "deltagam", "deltgam", "sk", "sigmakappa", "dz",
    "axo", "axid",     # Alpha Xi Delta
    "snoorpar",         # caught earlier
    # Multicultural / professional
    "akapsi", "psis", "psii",
}

GREEK_VOCAB = GREEK_LETTERS | FRAT_SORORITY_ABBREVIATIONS


# ---------------------------------------------------------------------------
# CUSTOM STOPWORDS (greatly expanded based on real output)
# ---------------------------------------------------------------------------
CUSTOM_STOPWORDS = {
    # site chrome remnants (in case regex misses some)
    "read", "more", "last", "post", "replies", "reply", "views", "view",
    "started", "ago", "year", "years", "month", "months",
    "day", "days", "hour", "hours", "minute", "minutes", "week", "weeks",
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec", "pm", "am",
    "pc",    # forum-nav term that showed up high in your output
    "pnm",   # "Potential New Member" abbreviation — generic
    "pnmlast", "spring", "fall", "gp", "grand", "prix", "semester", "self", "rank", "new", "curious", "kicked", "time", "upper", "lower", "mid", "low", "middle", "coming", "please", "wondering", "gp", "time", "poll",  # glued artifact

    # forum filler / contractions
    "im", "ive", "id", "dont", "doesnt", "didnt", "cant", "wont",
    "isnt", "wasnt", "arent", "werent", "youre", "youve", "youll",
    "theyre", "theyve", "theyll", "thats", "whats", "whos", "heres",
    "theres", "ill", "hes", "shes", "wouldnt", "couldnt", "shouldnt",
    "like", "just", "get", "got", "getting", "really", "know", "knew",
    "think", "thought", "would", "could", "should", "going", "go", "went",
    "one", "two", "three", "even", "still", "much", "well", "way", "ways",
    "also", "lot", "lots", "actually", "literally", "yeah", "okay", "ok",
    "thing", "things", "see", "seen", "say", "said", "make", "made",
    "take", "took", "taken", "want", "wants", "wanted", "need", "needs",
    "tell", "told", "good", "bad", "better", "best", "worst", "right",
    "wrong", "true", "false", "yes", "anyone", "everyone", "someone",
    "nobody", "something", "nothing", "anything", "everything", "let",
    "lets", "looking", "look", "looked", "next", "else", "first", "second",
    "though", "since", "ever", "never", "always", "every", "back", "come",
    "came", "give", "given", "gave", "find", "found", "feel", "felt",
    "put", "use", "used", "try", "tried", "trying", "many", "another",
    "around", "without", "within", "among", "etc",

    # --- Greek-life generic terms ---
    # These are real and meaningful, but so universal in this corpus they
    # crowd out everything specific. Filter them to surface the surprises.
    "pledge", "pledges", "pledging", "pledged",
    "house", "houses",
    "rush", "rushing", "rushed", "rushee", "rushees",
    "rushby",   # scraper artifact: "rush" + "by"
    "bid", "bids", "bidding",
    "chapter", "chapters",
    "frat", "frats", "fraternity", "fraternities",
    "sorority", "sororities",
    "brother", "brothers", "brotherhood",
    "sister", "sisters", "sisterhood",
    "greek", "greeks", "greekrank",
    "member", "members", "membership",
    "alumni", "alum", "alums",
    "recruitment", "recruit", "recruits", "recruited",
    "campus", "school", "schools", "university", "college",
    "purdue",  # they're all about Purdue — won't differentiate anything
    "guy", "guys", "girl", "girls", "dude", "dudes", "bro", "bros",
    "people", "person", "kid", "kids",
    "life",
    "function", "functions", "party", "parties", "social", "socials",
    "mixer", "mixers", "formal", "formals", "date", "dates", "tailgate",
    "tailgates", "pairing", "pairings",
    "year", "freshman", "sophomore", "junior", "senior",
    "love",  # commonly filler ("I love...") not informative
    "tier", "tiers", "ranking", "rankings", "ranked", "rank", "ranks",
    "top", "bottom", "middle",
    # slang variants of generic terms
    "srat",          # slang for sorority
    "bouse", "touse",   # slang for "the house" / "boilerhouse" — generic
    "func", "funcs",    # function abbreviation
    "sem",              # semester
    "real", "mid", "end", "lover", "using", "night", "idk",
    "ori",              # "orientation"? generic
    "space", "lazer", "benson", "ceti",   # likely names/jargon, low signal
    "yahu", "netanyahu",   # political mentions — interesting but filter from word counts
    "somehow", "weather", "christmas",   # context-dependent filler
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def is_greek_word(word: str) -> bool:
    """Check if a word is a Greek letter or a known chapter abbreviation."""
    if word in GREEK_VOCAB:
        return True
    for letter in GREEK_LETTERS:
        if word.startswith(letter) and len(word) > len(letter):
            rest = word[len(letter):]
            if rest in GREEK_VOCAB:
                return True
    return False


def clean_site_chrome(text: str) -> str:
    """
    Strip site-chrome phrases from raw text BEFORE tokenizing.

    Without this, things like "PledgeLast Post: 1 year ago" become the
    single token "pledgelast" after the tokenizer (because the regex
    [a-z]+ doesn't split on case changes — "PledgeLast" lowercases to
    "pledgelast" as one word). By replacing chrome with spaces first,
    "PledgeLast" becomes "Pledge " and tokenizes correctly.
    """
    for pattern in SITE_CHROME_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def tokenize(text: str, stopword_set: set[str]) -> list[str]:
    """Clean + lowercase + extract letter tokens + filter."""
    text = clean_site_chrome(text)
    text = text.lower()
    words = re.findall(r"[a-z]+", text)
    return [
        w for w in words
        if len(w) >= MIN_WORD_LENGTH
        and w not in stopword_set
        and len(set(w)) > 1
    ]


def split_greek_non_greek(words: list[str]) -> tuple[list[str], list[str]]:
    """Partition into (non_greek, greek) lists."""
    greek, non_greek = [], []
    for w in words:
        (greek if is_greek_word(w) else non_greek).append(w)
    return non_greek, greek


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}.")
    df = pd.read_csv(csv_path)
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    df["full_text"] = df["title"] + " " + df["body"]
    print(f"Loaded {len(df):,} rows from {csv_path}")
    return df


# ---------------------------------------------------------------------------
# CHAPTER-MENTION FILTER
# ---------------------------------------------------------------------------
def filter_posts_mentioning(df: pd.DataFrame, aliases: list[str]) -> pd.DataFrame:
    """
    Return only the rows whose text mentions any of the given aliases.

    Uses word-boundary regex so "aep" matches the word "aep" but NOT
    "aepi" or "aepidemic". This is important: without word boundaries,
    short abbreviations like "aep" would match inside unrelated words.
    """
    # Build one regex like: \b(aepi|ae pi|alpha epsilon pi|aep)\b
    # re.escape handles any special chars in aliases safely.
    pattern = r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b"
    mask = df["full_text"].str.contains(pattern, case=False, regex=True, na=False)
    matched = df[mask].copy()
    print(f"Posts mentioning {aliases[0].upper()}: {len(matched):,} "
          f"out of {len(df):,} total ({len(matched)/len(df)*100:.1f}%)")
    return matched


# ---------------------------------------------------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------------------------------------------------
def score_sentiment(df: pd.DataFrame, analyzer) -> pd.DataFrame:
    """
    Score each post's sentiment using VADER.

    VADER returns four scores for each text:
      - neg:      proportion of negative content (0 to 1)
      - neu:      proportion of neutral content (0 to 1)
      - pos:      proportion of positive content (0 to 1)
      - compound: overall score from -1 (most negative) to +1 (most positive)
                  This is the headline number — we focus on it.

    We add a "compound" column to the dataframe so we can sort, filter,
    and aggregate by sentiment later.

    A compound score interpretation:
      compound >=  0.05 → positive
      compound <= -0.05 → negative
      otherwise         → neutral
    These thresholds are VADER's official recommendation.
    """
    print(f"  Scoring sentiment for {len(df):,} posts...")
    df = df.copy()  # don't mutate the caller's dataframe

    # Apply polarity_scores to each post's full_text. The .apply() method
    # runs the given function on every row. We extract just the compound
    # score, then build a category label from it.
    scores = df["full_text"].apply(
        lambda t: analyzer.polarity_scores(str(t))["compound"]
    )
    df["sentiment"] = scores
    df["sentiment_label"] = df["sentiment"].apply(label_sentiment)

    return df


def label_sentiment(compound: float) -> str:
    """Bucket a compound score into positive / neutral / negative."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def plot_sentiment_distribution(df: pd.DataFrame, label: str,
                                output_path: Path):
    """
    Histogram + summary stats showing the sentiment distribution.

    This visual answers: "How positive/negative is this set of posts overall?"
    A histogram skewed right = mostly positive. Skewed left = mostly negative.
    Bimodal (two peaks) = polarized, with strong opinions on both sides.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # LEFT: histogram of compound scores
    ax1.hist(df["sentiment"], bins=40, color="steelblue", edgecolor="white")
    ax1.axvline(0, color="black", linestyle="--", linewidth=0.8, label="Neutral")
    ax1.axvline(df["sentiment"].mean(), color="red", linestyle="-",
                linewidth=1.5, label=f"Mean = {df['sentiment'].mean():.2f}")
    ax1.set_xlabel("Sentiment Score (compound)")
    ax1.set_ylabel("Number of Posts")
    ax1.set_title(f"Sentiment Distribution — {label}")
    ax1.set_xlim(-1, 1)
    ax1.legend()
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # RIGHT: bar chart of positive/neutral/negative counts
    counts = df["sentiment_label"].value_counts()
    # Reindex to enforce consistent ordering even if a category is empty
    counts = counts.reindex(["positive", "neutral", "negative"], fill_value=0)
    colors = ["seagreen", "gray", "indianred"]
    ax2.bar(counts.index, counts.values, color=colors)
    ax2.set_ylabel("Number of Posts")
    ax2.set_title(f"Positive / Neutral / Negative — {label}")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    # Add count labels on each bar
    for i, (cat, val) in enumerate(counts.items()):
        pct = val / len(df) * 100 if len(df) > 0 else 0
        ax2.text(i, val, f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom",
                 fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_sentiment_comparison(df_all: pd.DataFrame, df_focus: pd.DataFrame,
                              focus_label: str, output_path: Path):
    """
    Side-by-side comparison: is the focus chapter more positive/negative
    than the corpus overall?

    Two box plots make the comparison stark. Box plots show the median,
    quartiles, and outliers — much richer than just comparing averages.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [df_all["sentiment"], df_focus["sentiment"]]
    labels = [f"All Posts\n(n={len(df_all):,})",
              f"{focus_label} Mentions\n(n={len(df_focus):,})"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showmeans=True, widths=0.5)
    for patch, color in zip(bp["boxes"], ["steelblue", "darkorange"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Sentiment Score (compound)")
    ax.set_title(f"Sentiment Comparison: All Posts vs. {focus_label}")
    ax.set_ylim(-1.1, 1.1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Annotate means clearly
    ax.text(1, df_all["sentiment"].mean(),
            f"  μ={df_all['sentiment'].mean():.3f}", va="center", fontsize=10)
    ax.text(2, df_focus["sentiment"].mean(),
            f"  μ={df_focus['sentiment'].mean():.3f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def print_sentiment_summary(df: pd.DataFrame, label: str, n_examples: int = 3):
    """
    Print a sentiment summary to the terminal: distribution stats plus
    the most positive and most negative posts as examples.

    Showing the actual text alongside the scores helps you sanity-check
    that VADER is reading the posts the way you'd expect.
    """
    if len(df) == 0:
        return
    print(f"\n  --- Sentiment summary: {label} ---")
    print(f"  Mean compound score: {df['sentiment'].mean():+.3f}")
    print(f"  Median:              {df['sentiment'].median():+.3f}")
    counts = df["sentiment_label"].value_counts()
    for cat in ["positive", "neutral", "negative"]:
        n = counts.get(cat, 0)
        pct = n / len(df) * 100
        print(f"  {cat.capitalize():9s}: {n:>5,} posts ({pct:5.1f}%)")

    # Show some example posts at the extremes. Truncate body text so the
    # terminal doesn't get flooded.
    print(f"\n  Top {n_examples} MOST POSITIVE posts ({label}):")
    for _, row in df.nlargest(n_examples, "sentiment").iterrows():
        text = row["full_text"][:120].replace("\n", " ")
        print(f"    [{row['sentiment']:+.2f}] {text}...")

    print(f"\n  Top {n_examples} MOST NEGATIVE posts ({label}):")
    for _, row in df.nsmallest(n_examples, "sentiment").iterrows():
        text = row["full_text"][:120].replace("\n", " ")
        print(f"    [{row['sentiment']:+.2f}] {text}...")


# ---------------------------------------------------------------------------
# VISUALIZE
# ---------------------------------------------------------------------------
def plot_top_words(counter: Counter, n: int, title: str, output_path: Path,
                   bar_color: str = "steelblue"):
    top = counter.most_common(n)
    if not top:
        print(f"  [!] Nothing to plot for {title}")
        return
    words, counts = zip(*top)
    words, counts = words[::-1], counts[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.3)))
    ax.barh(words, counts, color=bar_color)
    ax.set_xlabel("Frequency")
    ax.set_title(title)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for i, count in enumerate(counts):
        ax.text(count, i, f" {count:,}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_side_by_side(non_greek_counter: Counter, greek_counter: Counter,
                      n: int, title: str, output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, n * 0.3)))
    for ax, counter, panel_title, color in [
        (ax1, non_greek_counter, f"Top {n} Topic Words", "steelblue"),
        (ax2, greek_counter, f"Top {n} Chapter / Greek Words", "darkorange"),
    ]:
        top = counter.most_common(n)
        if top:
            words, counts = zip(*top)
            ax.barh(words[::-1], counts[::-1], color=color)
        ax.set_title(panel_title)
        ax.set_xlabel("Frequency")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_wordcloud(counter: Counter, output_path: Path, colormap: str = "viridis"):
    if not counter:
        return
    wc = WordCloud(width=1600, height=900, background_color="white",
                   colormap=colormap, max_words=150
                   ).generate_from_frequencies(counter)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# ANALYSIS PIPELINE — runs the same flow on any subset of posts
# ---------------------------------------------------------------------------
def run_analysis(df: pd.DataFrame, stop_set: set[str], label: str,
                 output_subdir: Path, analyzer=None):
    """
    Run the full pipeline (word counts + sentiment) on a given DataFrame.

    If `analyzer` is provided, sentiment scoring is performed and an
    augmented dataframe (with sentiment columns) is returned. Otherwise
    sentiment is skipped and the original df is returned.
    """
    output_subdir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Analyzing: {label} ({len(df):,} posts) ---")

    if len(df) == 0:
        print("  [!] No posts to analyze, skipping.")
        return df

    # WORD ANALYSIS
    text = " ".join(df["full_text"].tolist())
    words = tokenize(text, stop_set)
    print(f"  Total words after cleaning: {len(words):,}")

    non_greek, greek = split_greek_non_greek(words)
    non_greek_counts = Counter(non_greek)
    greek_counts = Counter(greek)
    bigram_counts = Counter(" ".join(p) for p in ngrams(words, 2))

    print(f"\n  Top {TOP_N} NON-GREEK words ({label}):")
    for word, count in non_greek_counts.most_common(TOP_N):
        print(f"    {word:25s} {count:>6,}")

    print(f"\n  Top {TOP_N} GREEK / CHAPTER words ({label}):")
    for word, count in greek_counts.most_common(TOP_N):
        print(f"    {word:25s} {count:>6,}")

    print(f"\n  Top {TOP_N} BIGRAMS ({label}):")
    for phrase, count in bigram_counts.most_common(TOP_N):
        print(f"    {phrase:35s} {count:>6,}")

    # Save word data
    pd.DataFrame(non_greek_counts.most_common(),
                 columns=["word", "count"]).to_csv(
        output_subdir / "word_counts_non_greek.csv", index=False)
    pd.DataFrame(greek_counts.most_common(),
                 columns=["word", "count"]).to_csv(
        output_subdir / "word_counts_greek.csv", index=False)

    # Word charts
    plot_top_words(non_greek_counts, TOP_N,
                   f"Top {TOP_N} Topic Words — {label}",
                   output_subdir / "top_non_greek.png", "steelblue")
    plot_top_words(greek_counts, TOP_N,
                   f"Top {TOP_N} Chapter / Greek Words — {label}",
                   output_subdir / "top_greek.png", "darkorange")
    plot_top_words(bigram_counts, TOP_N,
                   f"Top {TOP_N} Bigrams — {label}",
                   output_subdir / "top_bigrams.png", "seagreen")
    plot_side_by_side(non_greek_counts, greek_counts, TOP_N,
                      f"GreekRank Analysis: {label}",
                      output_subdir / "comparison.png")
    make_wordcloud(non_greek_counts,
                   output_subdir / "wordcloud_non_greek.png", "viridis")
    make_wordcloud(greek_counts,
                   output_subdir / "wordcloud_greek.png", "plasma")

    # SENTIMENT ANALYSIS (if analyzer provided)
    if analyzer is not None:
        df = score_sentiment(df, analyzer)
        print_sentiment_summary(df, label)
        plot_sentiment_distribution(df, label,
                                    output_subdir / "sentiment_distribution.png")
        # Save the scored posts so you can explore them in Excel.
        df_to_save = df[["title", "sentiment", "sentiment_label", "body"]].copy()
        df_to_save.to_csv(output_subdir / "posts_with_sentiment.csv", index=False)
        print(f"  Saved: {output_subdir / 'posts_with_sentiment.csv'}")

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  GreekRank Analyzer v3.0")
    print("  (word frequency + chapter mentions + sentiment)")
    print("=" * 60)

    # NLTK first-time setup — stopwords + VADER lexicon
    try:
        stopwords.words("english")
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download("stopwords", quiet=True)
    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        print("Downloading VADER sentiment lexicon (one-time, ~125KB)...")
        nltk.download("vader_lexicon", quiet=True)

    stop_set = set(stopwords.words("english")) | CUSTOM_STOPWORDS
    analyzer = SentimentIntensityAnalyzer()
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataframe(INPUT_CSV)

    # 1. Full-corpus analysis (with sentiment)
    df_all_scored = run_analysis(df, stop_set,
                                 label="All Posts (Purdue)",
                                 output_subdir=OUTPUT_DIR / "all_posts",
                                 analyzer=analyzer)

    # 2. Chapter-focused analysis (with sentiment + comparison plot)
    if FOCUS_CHAPTER:
        chapter_df = filter_posts_mentioning(df, FOCUS_CHAPTER["aliases"])
        df_focus_scored = run_analysis(
            chapter_df, stop_set,
            label=f"Posts Mentioning {FOCUS_CHAPTER['name']}",
            output_subdir=OUTPUT_DIR / f"mentions_{FOCUS_CHAPTER['name'].lower()}",
            analyzer=analyzer,
        )

        # Cross-comparison: AEPi posts vs the whole corpus
        if len(df_focus_scored) > 0:
            plot_sentiment_comparison(
                df_all_scored, df_focus_scored,
                FOCUS_CHAPTER["name"],
                OUTPUT_DIR / f"sentiment_compare_{FOCUS_CHAPTER['name'].lower()}.png",
            )

    print(f"\nDone! All outputs in '{OUTPUT_DIR}/' (organized into subfolders).")


if __name__ == "__main__":
    main()

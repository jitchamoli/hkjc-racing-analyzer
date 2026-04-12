# HKJC Horse Racing Analyzer v4.0

Real-time horse racing analysis tool for the Hong Kong Jockey Club (HKJC). Fetches live race data, odds, and expert tips from racing.hkjc.com to generate betting recommendations using a combined rule-based + ML prediction system with 17 analytical factors, persistent horse database, and auto-learning pipeline.

## What's New in v4.0

- **17-factor analysis engine** - expanded from 14 to 17 scoring factors (added freshness, rating trend, season stakes)
- **QIN #1+#2 default strategy** - Quinella on model's rank 1 + rank 2 as default bet (hit 6/11 vs 2/11 for WIN alone)
- **BANKER WIN system** - 9 conviction signals (high score, rating surge, peak fitness, winning form, pace advantage, ideal draw, top jockey, proven earner, market backed); 7+ triggers BANKER WIN with "BET BIG" indicator
- **Market probability blending** - 70% model + 30% market-implied probabilities prevents underrating strong favourites
- **Market standout detection** - auto-promotes market favourite (35%+ implied) when it differs from model's top pick
- **Persistent horse database** - SQLite DB that accumulates horse, jockey, and trainer stats across race days
- **Auto-learning pipeline** - `--auto-learn` flag ingests results into horse DB and retrains ML model automatically
- **Expanded jockey database** - 25 jockeys with skill ratings (was 13)
- **Expanded trainer database** - 24 trainers with skill ratings (was 11)
- **Freshness scoring** - days since last run: 14-35 ideal, 7-14 quick backup, 35-60 fair break, 60-120 fitness query, 120+ layoff risk
- **Rating trend detection** - scores horses with rising (+5 or more) vs declining ratings
- **Season stakes factor** - rewards proven earners with high season prize money
- **WILDCARD BET section** - includes top outsider in exotic combinations for upset coverage
- **ONES TO WATCH** - renamed from "Dangers" for friendlier presentation
- **WhatsApp formatting** - bullet points, bold horse names, trimmed whitespace
- **Race Day Complete** - dashboard shows completion state after all races finished

## How It Works

### Data Sources (from racing.hkjc.com)

| Source | What It Provides |
|--------|-----------------|
| Race Card page | Horse entries, jockeys, trainers, draw, weight, gear, ratings, last 6 runs |
| Tips Index page | Live odds, Initial Tips Index, Race Day Tips Index, top picks per race |
| Odds page | Win and Place odds (when available on race day) |
| Horse Profile page | Running positions, sectional times, class history, C&D record |
| Barrier Trial page | Trial results for debut/Griffin horses |
| Results page | Finishing positions for completed races |
| Mark Six GraphQL API | Draw results, all-time number statistics, current jackpot |

### Prediction System

The analyzer combines two systems (50/50 weighted):

**Rule-Based System (50%) - 17 Factors**

| Factor | Weight | Description |
|--------|--------|-------------|
| HKJC Tips Index | 14% | Expert composite from racing media tipsters |
| Form | 12% | Last 6 race positions (supplemented by trial data for debut horses) |
| Rating | 9% | Official HKJC rating |
| Market Odds | 9% | Implied probability from live odds |
| Freshness | 6% | Days since last run (14-35 days = peak fitness) |
| Jockey | 6% | Jockey skill rating (25 riders: Moreira, Purton, Bowman, etc.) |
| Draw + Rail Bias | 5% | Barrier position adjusted for rail position, venue, and distance |
| Odds Movement | 5% | Steam moves (shortening) vs drifters (lengthening) |
| Weight Carried | 5% | Weight-to-rating ratio, field comparison, apprentice claims |
| Pace Map | 4% | Running style advantage based on field pace pressure |
| Sectional Speed | 4% | Finishing kick strength from last-400m times |
| Course & Distance | 4% | C&D win/place record at this venue and distance |
| Rating Trend | 4% | Rising ratings (+5 or more) vs declining |
| Trainer | 4% | Trainer success rate (24 trainers: A Cruz, J Size, F C Lor, etc.) |
| Class Movement | 3% | Advantage from class drop (easier field) or penalty for class rise |
| Gear Changes | 3% | Equipment impact (blinkers first time, tongue-tie, visor, etc.) |
| Season Stakes | 3% | Proven earners with high season prize money |

**ML-Based System (50%)**
- XGBoost classifier with 17 features matching all rule-based factors
- Auto-retrains when feature dimensions change or new results are ingested
- Persistent horse database feeds historical performance data

### Bet Type Logic

| Bet Type | When |
|----------|------|
| **BANKER WIN** | 7+ of 9 conviction signals fire — highest confidence, bet big |
| **WIN** | Very clear market standout (35%+ implied) or extreme model confidence |
| **QIN** (Quinella) | **Default** — rank 1 + rank 2 together (hit 6/11 races vs 2/11 for WIN alone) |
| **QNL PLACE** (Quinella Place) | Very tight top 2, extremely competitive field |
| **TRIO** | Top 3 horses extremely bunched, all viable winners |
| **PLACE** | Strong horse at short odds, safety play |
| **SKIP** | No positive EV or insufficient data |

### BANKER WIN System (9 Conviction Signals)

| Signal | Threshold |
|--------|-----------|
| High score | Rule score >= 68 |
| Rating surge | Rating change +5 or more |
| Peak fitness | 14-35 days since last run |
| Winning form | Form score >= 60 |
| Pace advantage | Pace score >= 70 |
| Ideal draw | Draw score >= 70 |
| Top jockey | Jockey score >= 82 |
| Proven earner | Season stakes >= $2M |
| Market backed | Implied probability >= 10% |

When 7+ signals fire, the horse is flagged as **BANKER WIN** with a gold "BET BIG" badge on the dashboard. BANKER overrides QIN default and auto-promotes to rank 1.

### Advanced Analytics

**Rail Position Draw Bias**
- Maps 3 venues (Sha Tin, Happy Valley) x 3 distance buckets (short/mid/long) x 8 rail positions
- Calculates ideal draw range and scores deviation from it
- Falls back to basic draw scoring when course info unavailable

**Pace Scenario Modeling**
- Classifies each horse as Leader/Stalker/Midfield/Closer from running positions or form heuristics
- Counts leaders in the field to determine pace pressure
- 0-1 leaders = easy lead (leaders advantaged), 3+ leaders = hot pace (closers advantaged)
- Short races (1200m) boost leaders, long races (1800m+) boost closers

**Odds Movement Tracking**
- SQLite database (WAL mode) stores timestamped odds snapshots
- Detects steam moves (30%+ shortening), strong support, stable markets, drifters
- Late movement detection: compares last quarter of snapshots to earlier readings
- Periodic polling support for race-day automation

**Class Movement Detection**
- Parses HKJC class hierarchy (Class 1-5 + Griffin)
- Big drop (2+ classes): 88 score | Single drop: 72 | Same: 50 | Rise: 38 | Big rise: 25

**Gear Change Impact**
- Scores common gear codes: B (blinkers), TT (tongue-tie), V (visor), H (hood), etc.
- First-time blinkers have highest strike rate among gear changes
- Blinkers off is treated as a mixed/negative signal

**Weight Analysis**
- Evaluates weight-to-rating ratio (light weight for high rating = advantage)
- Compares effective weight against field average
- Accounts for apprentice jockey weight claims (3-10 lbs)

## Project Structure

```
hkjc_racing_analyzer/
  hkjc_racing_analyzer.py   # Main analyzer (v4.0) - 17-factor engine + BANKER + QIN default
  hkjc_data_fetcher.py      # Real-time data layer (HKJC API + parser + new data fields)
  generate_predictions.py    # HTML dashboard generator + WhatsApp message formatter
  horse_database.py          # Persistent SQLite horse/jockey/trainer database
  results_fetcher.py         # Fetch race results + auto-learn pipeline (--auto-learn)
  advanced_analytics.py      # 8 specialized analyzers (draw bias, pace, sectional,
                             #   barrier trial, C&D, class movement, gear, weight)
  odds_tracker.py            # SQLite odds history + movement analyzer + polling
  ml_predictor.py            # XGBoost ML prediction module (17 features)
  betting_optimizer.py       # Kelly Criterion + value bet optimizer
  mark6_analyzer.py          # Mark Six lottery analyzer (GraphQL + frequency analysis)
  interactive_analyzer.py    # Interactive mode with manual odds input
  racing_scheduler.py        # Automated daily race analysis (cron)
  results_tracker.py         # Track predictions vs actual results
  run_workflow.sh            # Full workflow runner script
  setup_cron.sh              # Cron job setup for automation
  requirements.txt           # Python dependencies
  models/                    # Trained ML models (auto-generated)
  results/                   # Saved analysis results and HTML dashboards
  tracking_data/             # SQLite databases (odds history, horse history, predictions)
  training_data/             # Historical race data for ML training
  logs/                      # Application logs
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### Run the Racing Analyzer

```bash
python hkjc_racing_analyzer.py
```

You'll be prompted for:
1. **Race date** - enter `YYYY/MM/DD` or press Enter to auto-detect the next meeting
2. **Race number** - enter `1-11`, or `all` to analyze the full meeting

### Run the Mark Six Analyzer

```bash
python mark6_analyzer.py
```

### Example Output

```
  HKJC HORSE RACING ANALYZER v1.0.0 - REAL-TIME DATA EDITION
  ============================================================

  PREDICTION METHODOLOGY (v3.0 - Full 14-Factor Analytics)
  --------------------------------------------------------
  RULE-BASED (50%):  Rating 12% | Form 12% | Tips 12% | Odds 9% | Jockey 7%
                     Trainer 5% | Draw+Rail 6% | Pace 6% | C&D 6% | Odds Move 6%
                     Sectional 5% | Class Move 5% | Gear 4% | Weight 5%
  ML-BASED   (50%):  XGBoost classifier (14 features)

  RACE 5 - Sha Tin
  Date: 2026/04/12 | Distance: 1200m | Class: Class 4

  Rank No  Horse               Jockey          Odds    Comb%   EV%     Bet
  1    4   CALL ME SPARKLE     M L Yeung       $3.8    55.4%   +111%   WIN
  2    3   CONRAD PATCH        H Bowman        $9.1    46.4%   +322%   WIN
  ...
```

## Requirements

- Python 3.10+
- Internet connection (fetches live data from racing.hkjc.com)
- Playwright + Chromium (optional, for JS-rendered odds pages)

## Key Dependencies

- `xgboost` - ML prediction engine
- `scikit-learn` - Model training and evaluation
- `numpy` - Numerical computations
- `playwright` - Browser automation for JS-rendered pages
- `beautifulsoup4` - HTML parsing
- `requests` - HTTP requests

## Disclaimer

For entertainment and educational purposes only. Not financial advice. Past performance does not guarantee future results. Gamble responsibly.

# HKJC Horse Racing Analyzer v3.0

Real-time horse racing analysis tool for the Hong Kong Jockey Club (HKJC). Fetches live race data, odds, and expert tips from racing.hkjc.com to generate betting recommendations using a combined rule-based + ML prediction system with 14 analytical factors.

## What's New in v3.0

- **14-factor analysis engine** - expanded from 7 to 14 scoring factors for deeper race analysis
- **Rail position draw bias** - adjusts draw advantage based on rail position (A, A+1, A+2, A+3, B, B+2, C, C+3) per venue and distance
- **Pace map analysis** - classifies running styles (Leader/Stalker/Midfield/Closer) and models pace pressure scenarios
- **Sectional time analysis** - evaluates finishing speed from last-400m sectional benchmarks
- **Barrier trial analysis** - supplements form scores for debut/Griffin horses with trial performance data
- **Course & Distance form** - tracks C&D win/place record with venue and distance matching (100m tolerance)
- **Odds movement tracking** - SQLite-backed odds history with steam move and drifter detection
- **Class drop/rise detection** - scores advantage when a horse drops or rises in class
- **Gear change analysis** - evaluates equipment changes (blinkers, tongue-tie, visor, hood, etc.)
- **Weight analysis** - scores weight-to-rating ratio, field comparison, and apprentice claims
- **Dynamic weight redistribution** - when a factor has no data, its weight is redistributed proportionally
- **HTML dashboard output** - beautiful dark-themed betting strategy dashboard
- **XGBoost with 14 features** - ML model extended to match all analytical factors

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

**Rule-Based System (50%) - 14 Factors**

| Factor | Weight | Description |
|--------|--------|-------------|
| Rating | 12% | Official HKJC rating |
| Form | 12% | Last 6 race positions (supplemented by trial data for debut horses) |
| HKJC Tips Index | 12% | Expert composite from racing media tipsters |
| Market Odds | 9% | Implied probability from live odds |
| Jockey | 7% | Jockey skill rating (Moreira, Purton, Teetan, Bowman, etc.) |
| Draw + Rail Bias | 6% | Barrier position adjusted for rail position, venue, and distance |
| Pace Map | 6% | Running style advantage based on field pace pressure |
| Course & Distance | 6% | C&D win/place record at this venue and distance |
| Odds Movement | 6% | Steam moves (shortening) vs drifters (lengthening) |
| Trainer | 5% | Trainer success rate (J Size, A Cruz, F C Lor, etc.) |
| Sectional Speed | 5% | Finishing kick strength from last-400m times |
| Class Movement | 5% | Advantage from class drop (easier field) or penalty for class rise |
| Weight Carried | 5% | Weight-to-rating ratio, field comparison, apprentice claims |
| Gear Changes | 4% | Equipment impact (blinkers first time, tongue-tie, visor, etc.) |

**ML-Based System (50%)**
- XGBoost classifier with 14 features matching all rule-based factors
- Auto-retrains when feature dimensions change
- Synthetic training data with realistic HKJC distributions

### Bet Type Logic

| Bet Type | When |
|----------|------|
| **WIN** | Clear standout horse with positive expected value |
| **PLACE** | Strong horse but odds too short for WIN value |
| **QIN** (Quinella) | Top 2 horses clearly separated from the field |
| **QNL PLACE** (Quinella Place) | Tight competitive field, top horses bunched |
| **TRIO** | Top 3 horses identified with strong combined confidence |
| **SKIP** | No positive EV, wide-open field, or insufficient data |

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
  hkjc_racing_analyzer.py   # Main analyzer (v3.0) - entry point
  hkjc_data_fetcher.py      # Real-time data layer (HKJC API + parser)
  advanced_analytics.py      # 8 specialized analyzers (draw bias, pace, sectional,
                             #   barrier trial, C&D, class movement, gear, weight)
  odds_tracker.py            # SQLite odds history + movement analyzer + polling
  ml_predictor.py            # XGBoost ML prediction module (14 features)
  betting_optimizer.py       # Kelly Criterion + value bet optimizer
  mark6_analyzer.py          # Mark Six lottery analyzer (GraphQL + frequency analysis)
  interactive_analyzer.py    # Interactive mode with manual odds input
  racing_scheduler.py        # Automated daily race analysis (cron)
  results_tracker.py         # Track predictions vs actual results
  odds_scraper.py            # Legacy odds scraper (superseded by data fetcher)
  screenshot_capture.py      # Screenshot capture utility
  update_results.py          # Post-race result updater
  run_workflow.sh            # Full workflow runner script
  setup_cron.sh              # Cron job setup for automation
  requirements.txt           # Python dependencies
  models/                    # Trained ML models (auto-generated)
  results/                   # Saved analysis results and HTML dashboards
  tracking_data/             # SQLite odds history + prediction tracking
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

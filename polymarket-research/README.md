# Polymarket Steady Climbers

Research into Polymarket traders whose profit chart climbs steadily, what they
actually trade, and a boring strategy that survived testing on the whole market.
Data through 19 Nov 2026. Full write-up with charts: `report/steady-climbers.html`.

## Findings

1. **Steady curves are rare.** 149,451 wallets scanned from 25 leaderboard lists;
   **543** pass every "beautiful chart" rule over the last 12 months (≥ $3k profit,
   ≥ 150 days of climbing, still trading, R² ≥ 0.8 to a straight line, ≥ 70% of
   months up, drawdown ≤ 25% of profit, no lumpy single days or months).
2. **What they do** (956,239 of their positions, every market tagged):

   | Style | Traders | Median 1Y profit | Profit per $ traded | Typical entry |
   |---|---|---|---|---|
   | High-probability "bond" buyer | 176 | $22.5k | 1.9% | 92.5¢ |
   | Market maker / liquidity provider | 155 | $55.6k | 1.7% | 69¢ |
   | Discretionary / event trader | 100 | $37.8k | 6.8% | 72¢ |
   | Sports / esports | 67 | $49.5k | 6.8% | 58¢ |
   | Weather | 22 | $10.1k | 7.3% | 58¢ |
   | Crypto up/down bot | 19 | $16.6k | 3.3% | 78¢ |

3. **Steadiness persists.** Chosen on the first half of the year only, 81% of steady
   traders were profitable in the second half, against 52% of equally profitable
   bumpy traders (1.6% vs 7.3% lost more than their first-half gain).
4. **The obvious shortcuts lose money** (all 132,586 markets resolved in the year,
   realistic fills, fee schedule from 20 Nov 2026):
   - Buying favourites at 90–99¢ loses 0.7–3.5% per trade. Favourites win *less*
     often than their price implies (95–97¢ band: won 94.3%, needed 96.3%).
   - Copying the steady wallets one hour after they buy: −0.38% per trade
     (worse with longer delays).
   - Stop-losses lower returns at every level tested.
5. **How the bond buyers really win:** about a third of their money goes in *after*
   an event's scheduled end once the result is known (resolution lag; needs a
   human check of the official source), and the largest slice is long-dated
   "nothing ever happens" bets on deadline markets.

## The strategy: Deadline Fade

Bet that "Will X happen by <date>?" deadlines pass quietly.

- **Markets:** single yes/no deadline questions with ≥ $25k volume in politics,
  geopolitics, economics, finance, tech, culture or crypto price. Skip sports,
  esports, weather, crypto up/down and multi-outcome (negRisk) events.
- **Entry:** the likely side (usually No) at 95–98¢ with 7–30 days to the deadline
  and a book spread ≤ 2¢. Read the market's resolution rules first.
- **Order:** limit order one tick under the ask, never above the midpoint (makers
  pay no fee).
- **Size:** 0.5% of the account per market, ≤ 1.5% per event, ~5% per story.
- **Exit:** hold to resolution; no stop-loss.

Backtest (Nov 2025 – Nov 2026, 1,834 qualifying trades, median of 60 runs with
random same-day order, $10k start):

| Execution | Per year | Max drawdown | R² of equity curve | Trades won |
|---|---|---|---|---|
| Limit orders (maker) | +8.9% | 2.3% | 0.92 | 97.8% |
| Market orders (taker) | +6.4% | 2.3% | 0.87 | 97.8% |

It was positive in each half of the year tested alone (+7.2%/yr and +9.0%/yr);
10 of 13 months were up and the worst month was −0.8%. Bigger positions (2% each)
did worse, because a small account can then only take part of each burst of
opportunities. Expect roughly 98% wins and about +1% per trade; one loss erases
about 27 wins.

**Stop and re-check** if the account falls 6% from its peak or the last 100
trades average below −4% (the worst stretch in testing was −3.5%).

## Daily use

```bash
pip install requests pandas numpy pyarrow
python3 scanner.py 10000      # your account size in USD; read-only, places no orders
```

The scanner lists today's qualifying markets with a suggested limit price and
stake, and writes `report/scanner_today.csv`.

## Reproducing the research

All endpoints are public (Polymarket Data API v2, Gamma, CLOB). Raw data goes to
`data/` (about 6 GB, git-ignored); small summaries are in `results/` and `report/`.

| Step | Script | What it does |
|---|---|---|
| 1 | `01_candidates.py`, `01b_extend_candidates.py` | wallet pool from leaderboards |
| 2 | `02_fetch_pnl.py` (`data/candidates.csv` then `data/candidates_ext.csv ext`) | daily P&L history per wallet |
| 3 | `03_score_curves.py`, `03b_select.py` | curve metrics (`curve_metrics.py`) and the 543 (`selection.py`) |
| 4 | `04_fetch_trader_detail.py`, then `… markets` | positions, trades, activity, market metadata |
| 5 | `05_features.py`, `07_archetypes.py`, `08_bond_deepdive.py` | strategy fingerprints and styles |
| 6 | `06_persistence.py`, `06b_is_steady.py` | out-of-sample persistence |
| 7 | `09a`–`09e` | market universe, hourly prices, favourite backtest, entry conditions |
| 8 | `09f`, `09g`, `09j`, `09l`, `09n`, `09m` | deadline markets, final rule, exits, Monte Carlo portfolio |
| 9 | `09h`, `09i` | copy-trading tests |
| 10 | `09k` | liquidity-reward snapshot |
| 11 | `10_report_assets.py`, `11_report_data.py` | report data (`report/template.html` + JSON → page) |

Lists: `report/steady_climbers_543.csv` (all 543 with style and stats) and
`report/steady_bond_buyers.csv`. Chart grids: `results/charts/`.

## Caveats

- One year of history; the edge is small (about +1% per trade) and could fade.
- Maker results assume limit orders fill near the midpoint; taker results are the
  conservative bound.
- Fee figures for 20 Nov 2026 come from search extracts of Polymarket's docs;
  check the live fee page.
- Polymarket's international site is not available to US residents. Not financial
  advice.

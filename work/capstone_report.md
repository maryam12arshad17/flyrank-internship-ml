# Capstone Report — Refresh / Content Opportunity Scoring
- **Author:** Maryam Arshad
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/maryam12arshad17/flyrank-internship-ml.git
- **Date:** 31-08-2026

## 0. Abstract
This project asks which content pages on FlyRank's client portfolio are worth
reviewing first, using March 2026 (month=2026-03) daily search performance
data from `fact_content_daily_performance`. A transparent CTR-vs-position
rule baseline was built first, then compared to a Random Forest classifier
predicting whether a page's impressions declined within the month. Evaluated
on a client-grouped, out-of-sample split, the Random Forest showed
directional improvement over the rule baseline (precision@50 = 0.56 vs.
0.16), though a naive random split inflated this to 0.84 — revealing real
client-specific memorization that the honest split corrects for. The output
is a ranked, reason-coded action queue intended as decision-support for a
content team's monthly review, not an automated or guaranteed outcome.

## 1. Problem framing
**Decision supported:** which content pages a reviewer should look at first
each month, out of thousands, when time is limited.
**Unit of analysis:** one content page (`content_hash_id`), for one client
(`client_hash_id`), aggregated over one calendar month.
**Output:** a ranked queue with a decline-risk score, a human-readable
reason code, and a suggested action (e.g. `review_and_refresh`,
`monitor_only`).
**Action a human takes:** manually reviews the top-ranked pages, checking
the live page and current SERP behavior before deciding to refresh content,
fix a title/snippet, or leave it as-is.
**Cost of a wrong call:** a false positive wastes reviewer time on a page
that didn't need attention; a false negative lets a genuinely declining page
go unreviewed for another month. Neither is catastrophic, which is why this
is framed as decision-support (a ranked queue for a human), not an automated
action.
**Why ML helps:** with tens of thousands of pages per month and only human
review time available, a transparent rule alone (the baseline) captures one
pattern (CTR-vs-position) but misses others; a model can combine more
signals into a single rankable score, provided it's validated honestly.

## 2. Data safety
**Data used:** `fact_content_daily_performance` (daily grain, aggregated to
monthly per content item) for `month=2026-03`, joined where needed to
`dim_content` and `dim_clients` for context. Features: `gsc_impressions_total`,
`gsc_clicks_total`, `gsc_avg_position`, `ga4_sessions_total`,
`ga4_pageviews_total`, `ga4_engaged_sessions_total`, `days_observed`,
`days_gsc_available`, and three engineered ratios (`ctr`, `engagement_rate`,
`gsc_coverage`).

**Deliberately excluded:**
- `impression_change` — the exact quantity the `is_declining` label is
  derived from; confirmed via a deliberate-leak test (AUC jumped to 1.0 when
  included, Section 5).
- `last_optimized_date`, `optimization_eligible_date` — product/decision
  flags representing FlyRank's own past actions on the content; using these
  would mean learning an old system's decisions, not the world.
- AI-referral columns (`ai_chatgpt`, `ai_perplexity`, etc.) — present in
  ~69% of rows but non-zero in only 0.03% of them; too sparse to carry
  signal for this lane.
- `client_hash_id`, `content_hash_id`, `report_date` — used only for
  grouping/joining, never as model features (would let the model memorize
  identifiers instead of learning generalizable patterns).

**Leakage risks considered:** future-window leakage (content-update dates
that postdate the analysis month were treated as "not yet updated by March,"
never as a numeric days-since value that could imply future knowledge);
label-derived features (tested and excluded, see Section 5); pseudonymous
IDs used only for grouping. No client names, domains, URLs, or raw exports
appear anywhere in `work/`.

## 3. Baseline
**The rule:** flag a page if it holds a good search position (top 10 average
position) but its CTR falls below the average CTR of other top-10 pages —
a likely title/snippet problem rather than a ranking problem. Score =
flag × impressions, so larger-visibility opportunities rank first.

**Why it's a fair comparison:** it is a transparent, hand-written rule with
no fitted weights, built on the same March 2026 slice, and evaluated with
the same precision@50 metric later used for the model.

**Signal testing behind the rule (from the signal audit):** staleness
(content_updated_date) was tested first and found **OPPOSITE** of the
assumption — unupdated content outperformed recently-updated content, so
staleness was dropped from the rule. CTR-vs-position was tested and found
**CONFIRMED** (top-10 position content showed roughly double the CTR of
lower-ranked content), so the final rule is built on this signal instead.

**Baseline numbers:** precision@50 = 0.16 on the `is_declining` label
(evaluated later for direct model comparison) — notably below the base rate
of that label (~0.18), since the rule was built for a different target
(CTR-fix opportunities), not decline prediction. This mismatch is itself an
honest finding, not a flaw.

## 4. Model / analysis
**Method:** Random Forest Classifier (`n_estimators=200`, `max_depth=6`),
chosen per the toolkit's guidance for a yes/no question with an observed
label — started from Logistic Regression as the readable baseline, moved to
Random Forest for a stronger, still-interpretable comparison (via feature
importances).

**Feature list (11):** `gsc_impressions_total`, `gsc_clicks_total`,
`gsc_avg_position`, `ga4_sessions_total`, `ga4_pageviews_total`,
`ga4_engaged_sessions_total`, `days_observed`, `days_gsc_available`, `ctr`,
`engagement_rate`, `gsc_coverage`. All are trailing, past-only aggregates
over the full month — none touch data past the decision moment.

**Deliberately left out:** see Section 2's exclusion list — label-derived
and product-flag columns.

**Target / proxy definition:** `is_declining` = 1 if a content item's GSC
impressions in the second half of March (Mar 16–31) were lower than the
first half (Mar 1–15), else 0 — a within-month, past-observable proxy for
decline.

## 5. Evaluation
**Split:** client-grouped (`GroupShuffleSplit`, 70/30, by `client_hash_id`).
A random row-level split would let the same client's content appear in both
train and test, letting the model partly memorize client-specific patterns.
The grouped split asks the honest question: does this generalize to a
client the model has never seen?

**Model vs. baseline, same split, same metric (precision@50):**

| Method | precision@50 | AUC | Base rate |
|---|---|---|---|
| Base rate (random ranking) | 0.18 | 0.50 | 0.18 |
| Week-4 rule baseline | 0.16 | — | 0.18 |
| Logistic Regression | 0.66 | 0.775 | 0.18 |
| Random Forest | 0.58 | 0.803 | 0.18 |

**Random vs. grouped split, same model (Random Forest):**

| Split | AUC | precision@50 |
|---|---|---|
| Random (naive) | 0.861 | 0.84 |
| Grouped by client (honest) | 0.804 | 0.56 |

The gap (0.84 → 0.56) shows a naive split overstated performance —
part of the "before" score reflected client-specific memorization, not
generalizable signal. **0.56 is the number this report trusts.**

**Leakage audit:** adding the label-derived `impression_change` column
pushed AUC to a perfect 1.0, confirming the test harness catches leakage and
confirming why that column stays excluded. A single-feature AUC scan found
no individual feature above 0.95 (highest: `days_gsc_available` at 0.814,
plausible rather than suspicious).

**Error analysis:** of 49,823 grouped-split test predictions, 8,987 (18%)
were wrong. Errors clustered around content with thin GSC coverage (under
50% of days in the month), where predictions are inherently less reliable —
but at least one wrong case had full data coverage, showing aggregate
monthly stats alone don't catch every decline pattern.

## 6. Interpretation
**Top features (Random Forest importances):** `days_gsc_available` (0.30)
and `gsc_coverage` (0.23) dominate, followed by `gsc_impressions_total`
(0.18) and `gsc_avg_position` (0.15). This is a caution, not a clean win:
since the label compares first-half vs. second-half impressions, content
whose GSC data simply stops mid-month can look "declining" for a
data-availability reason rather than a genuine performance reason — flagged
explicitly rather than treated as clean signal.

**Negative / surprising results:** the staleness signal was **OPPOSITE** of
assumption (Section 3) — unupdated content outperformed recently-updated
content, likely because unupdated pages are older, established, proven
performers. A search-volume "quick win" signal tested separately was also
**OPPOSITE**: high-search-volume keywords averaged fewer clicks than
low-volume ones, likely due to competition eating into the larger opportunity.
Both are reported as valid findings, not discarded.

## 7. Recommendation
The final ranked queue flags content using percentile-based thresholds (90th
percentile of predicted decline probability, not a fixed cutoff) with four
reason codes: `high_risk_ranked_page` (highest priority — good position,
high decline risk), `high_risk_low_visibility`, `thin_data_coverage` (flag
requires data verification before action), and `lower_priority` (monitor
only). All 4,988 high-risk flagged rows have reliable GSC coverage (≥50%),
so a reviewer can trust the top of the queue directly.

**How an editor uses this tomorrow:** open the top-ranked rows, verify the
live page and current SERP behavior, then decide whether to refresh content
or fix title/snippet — never auto-publish or auto-edit from the `action`
label alone. Rows in `thin_data_coverage` (58% of the March queue) should be
treated as flags to investigate the data gap first, not action-ready
recommendations.

**Confidence and limits stated explicitly:** this queue reflects March 2026
only, on this 104-client cohort; performance on new clients or new months is
unvalidated until re-run. This is observed, decision-support evidence — not
a guarantee that acting on a flagged page improves its performance.

## 8. Reproducibility
**Re-run from a fresh clone:**
1. Open `work/notebooks/w05_model.ipynb` (or the capstone notebook) in
   Google Colab.
2. Add a Colab Secret named `HF_TOKEN` (a Hugging Face **Read**-type token,
   with dataset access granted at
   `huggingface.co/datasets/FlyRank/internship-warehouse`).
3. Runtime → Run all.

**Random seed:** `random_state=42` fixed throughout (train/test split and
Random Forest).
**Environment:** `duckdb`, `pandas`, `scikit-learn` (standard Colab
versions as of 2026-08); no pinned requirements file beyond what Colab
provides by default.
**Sealed/holdout claims:** none made — this report's precision@50 = 0.56
figure is a grouped-split validation number, not a sealed final-month
evaluation; `month=2026-06` was not touched for any part of this analysis.

## 9. Acknowledgments & data credit
Built on the FlyRank ML Internship dataset. Learn more at
[flyrank.ai](https://flyrank.ai).

---
> **Claims checklist:** observed / measured / directional / decision-support
> language used throughout (Sections 0, 1, 6, 7). Base rate (0.18) reported
> next to precision@50 in Section 5. No causal claims. No client-identifying
> details anywhere. No claim of predicting Google's algorithm.ying details · numbers in this report
> match a fresh re-run.

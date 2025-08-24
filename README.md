# Amazon SP Strategy Agent (MVP)

This is a minimal, single-run agent that reads:
1) **Sponsored_Product_Search_term_report** (Search Term performance)
2) **Sponsored_Product_Search_term_impression_share_report** (Search Term Impression Share)

It merges both and generates **actionable recommendations** to optimize toward your target ACOS (e.g., ROAS 5 → ACOS 20%).

## How to Use

1. Export the two raw CSVs from Amazon Ads Console (SP Search Term & SP Search Term Impression Share).
2. Place them anywhere on your machine.
3. Create a Python venv and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
   pip install -r requirements.txt
   ```
4. Edit `config.yaml` to match your goals (target ACOS, brand terms, thresholds, etc.).
5. Run:
   ```bash
   python agent/main.py --search_csv "/path/to/SearchTerm.csv"                         --is_csv "/path/to/ImpressionShare.csv"                         --out_csv "./agent_recommendations.csv"                         --config "./config.yaml"
   ```

## Output

- `agent_recommendations.csv`: one row per **search term**, with:
  - `action`: one of
    - `bid_increase`, `bid_decrease`, `harvest_to_exact`, `add_negative_exact`, `maintain`
  - `suggested_bid`: a recommended bid (if applicable)
  - `reason`: short rationale
  - plus key metrics (`acos`, `roas`, `cpc`, `cvr`, `impression_share`, etc.)

## Notes

- This MVP uses **rules + Bayesian smoothing** for CVR and calculates a **max CPC** to hit your target ACOS:
  - `expected_revenue_per_click = CVR * ASP`
  - `ACOS ≈ CPC / (CVR * ASP)` → `max_CPC = target_ACOS * CVR * ASP`
- If your CSV column names differ slightly, the agent includes a flexible column-normalization step.
- Start simple with rules, then evolve into ML-based predictions as you collect data.

#!/usr/bin/env python
import argparse, pandas as pd, numpy as np, yaml, re, math, sys, os
from typing import Dict, List, Tuple

# ---------- Column normalization for Search Term report ----------
SEARCH_TERM_COLMAP = {
    # canonical : list of possible variants
    "start_date": ["start date"],
    "end_Date": ["end date"],
    "campaign_name": ["campaign name"],
    "ad_group_name": ["ad group name"],
    "keyword": ["keyword", "targeting", "keyword or product targeting"],
    "match_type": ["match type"],
    "search_term": ["customer search term", "search term"],
    "impressions": ["impressions"],
    "clicks": ["clicks"],
    "ctr": ["click-thru rate (ctr)", "ctr"],
    "cpc": ["cost per click (cpc)", "cpc"],
    "spend": ["spend", "cost"],
    "sales": ["7 day total sales", "attributed sales 7d", "14 day total sales", "30 day total sales", "sales"],
    "orders": ["7 day total orders (#)", "attributed units ordered 7d", "orders"],
    "acos": ["total advertising cost of sales (acos)", "acos"],
    "roas": ["total return on advertising spend (roas)", "roas"]
}

# ---------- Column normalization for Impression Share report ----------
IS_COLMAP = {
    "start_date": ["Start Date"],
    "end_Date": ["End Date"],
    "search_term": ["search term", "search term (exact)", "customer search term"],
    "marketplace": ["marketplace"],
    "impression_share": ["impression share", "search term impression share"],
    "rank": ["rank", "search term rank", "top of search rank"]
}

def normalize_columns(df: pd.DataFrame, colmap: Dict[str, List[str]]) -> pd.DataFrame:
    lower_cols = {c.lower().lstrip().rstrip(): c for c in df.columns}
    mapping = {}
    for canonical, variants in colmap.items():
        for v in variants:
            if v in lower_cols:
                mapping[lower_cols[v]] = canonical
                break
    # keep unknowns as-is (lowercased)
    out = df.copy()
    out.columns = [mapping.get(c, c.lower().lstrip().rstrip()) for c in df.columns]
    return out

def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    s = s.replace(",", "").replace("$","")
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def safe_div(a, b):
    try:
        if b == 0 or pd.isna(a) or pd.isna(b):
            return 0.0
        return float(a) / float(b)
    except:
        return 0.0

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def detect_brand_term(term: str, brand_terms: List[str]) -> bool:
    if not isinstance(term, str):
        return False
    t = term.lower()
    return any(bt in t for bt in brand_terms)

def bayesian_cvr(orders, clicks, prior_clicks, prior_cvr):
    # posterior = (orders + prior_cvr * prior_clicks) / (clicks + prior_clicks)
    return safe_div(orders + prior_cvr * prior_clicks, clicks + prior_clicks)

def estimate_asp(sales, orders, global_fallback_asp):
    asp = safe_div(sales, max(orders, 1e-9))
    if asp <= 0:
        return global_fallback_asp
    return asp

def suggest_bid(current_cpc, target_acos, cvr, asp, min_bid, max_bid, direction=None, pct_change=None):
    # Max CPC allowed to hit target ACOS: ACOS ≈ CPC / (CVR * ASP) -> CPC_max = target_ACOS * CVR * ASP
    cap = target_acos * cvr * asp
    if direction == "increase":
        new_bid = current_cpc * (1 + (pct_change or 0.1))
    elif direction == "decrease":
        new_bid = current_cpc * (1 - (pct_change or 0.1))
    else:
        new_bid = current_cpc
    # Clip by cap and global bounds
    if not np.isnan(cap) and cap > 0:
        new_bid = min(new_bid, cap)
    new_bid = max(min_bid, min(max_bid, new_bid if new_bid>0 else min_bid))
    return round(new_bid, 4), cap

def main(args):
    cfg = load_config(args.config)

    # Load data
    st_df = pd.read_csv(args.search_csv, dtype=str, encoding="utf-8", na_filter=False)
    is_df = pd.read_csv(args.is_csv, dtype=str, encoding="utf-8", na_filter=False)

    st_df = normalize_columns(st_df, SEARCH_TERM_COLMAP)
    is_df = normalize_columns(is_df, IS_COLMAP)

    # Coerce numerics
    num_cols_st = ["impressions", "clicks", "cpc", "spend", "sales", "orders", "acos", "roas"]
    for c in num_cols_st:
        if c in st_df.columns:
            st_df[c] = st_df[c].apply(to_float)
    # standardize term key
    if "search_term" not in st_df.columns:
        raise ValueError("Search Term report missing a 'search_term' column after normalization.")

    # Aggregate Search Term performance over the file range
    perf = st_df.groupby("search_term", as_index=False).agg({
        "impressions": "sum",
        "clicks": "sum",
        "spend": "sum",
        "sales": "sum",
        "orders": "sum",
        "cpc": "mean"  # average CPC
    })
    # derived metrics
    perf["ctr"] = perf.apply(lambda r: safe_div(r["clicks"], r["impressions"]), axis=1)
    perf["cvr"] = perf.apply(lambda r: safe_div(r["orders"], r["clicks"]), axis=1)
    perf["acos"] = perf.apply(lambda r: safe_div(r["spend"], r["sales"]), axis=1)
    perf["roas"] = perf.apply(lambda r: safe_div(r["sales"], r["spend"]), axis=1)

    # global ASP (fallback)
    total_sales = perf["sales"].sum()
    total_orders = perf["orders"].sum()
    global_asp = safe_div(total_sales, total_orders if total_orders>0 else 1e-9)

    # Impression Share data
    for c in ["impression_share"]:
        if c in is_df.columns:
            is_df[c] = is_df[c].apply(to_float)
    if "search_term" not in is_df.columns:
        raise ValueError("Impression Share report missing a 'search_term' column after normalization.")

    is_agg = is_df.groupby("search_term", as_index=False).agg({
        "impression_share": "mean",
    })

    merged = perf.merge(is_agg, on="search_term", how="left")

    # Classification + recommendations
    brand_terms = [t.lower() for t in cfg.get("brand_terms", [])]
    target_acos = float(cfg["target_acos"])
    min_clicks = int(cfg["min_clicks_for_decision"])
    min_orders_for_promo = int(cfg["min_orders_for_promo"])
    inc_pct = float(cfg["increase_bid_pct"])
    dec_pct = float(cfg["decrease_bid_pct"])
    min_bid = float(cfg["min_bid"])
    max_bid = float(cfg["max_bid"])
    prior_clicks = int(cfg["bayes_prior_clicks"])
    prior_cvr = float(cfg["bayes_prior_cvr"])
    neg_spend_thr = float(cfg["negation_spend_threshold"])
    harvest_acos_thr = float(cfg["harvest_exact_threshold_acos"])
    brand_is_floor = float(cfg["brand_is_floor"])
    generic_is_floor = float(cfg["generic_is_floor"])

    recs = []
    for _, r in merged.iterrows():
        term = r["search_term"]
        clicks = r.get("clicks", 0) or 0.0
        orders = r.get("orders", 0) or 0.0
        spend = r.get("spend", 0) or 0.0
        sales = r.get("sales", 0) or 0.0
        cpc = r.get("cpc", np.nan)
        acos = safe_div(spend, sales)
        roas = safe_div(sales, spend)
        is_share = r.get("impression_share", np.nan)
        tos_is = r.get("top_of_search_is", np.nan)

        is_brand = detect_brand_term(term, brand_terms)
        cvr_bayes = bayesian_cvr(orders, clicks, prior_clicks, prior_cvr)
        asp = estimate_asp(sales, orders, global_asp)

        # default action
        action = "maintain"
        suggested_bid = ""
        reason = []

        # Negation rule: spend high & no orders
        if clicks >= min_clicks and orders == 0 and spend >= neg_spend_thr:
            action = "add_negative_exact"
            reason.append(f"Clicks≥{min_clicks} & Orders=0 & Spend≥{neg_spend_thr:.2f}")
        else:
            # Harvest rule: good ACOS -> create Exact
            if orders >= min_orders_for_promo and acos > 0 and acos <= harvest_acos_thr:
                action = "harvest_to_exact"
                reason.append(f"Orders≥{min_orders_for_promo} & ACOS≤{harvest_acos_thr:.2f}")

            # Bid optimization
            # Growth if ROAS good and Impression Share low
            floor = brand_is_floor if is_brand else generic_is_floor
            if roas >= (1.0/target_acos) and (np.isnan(is_share) or is_share < floor):
                # increase bid up to cap
                new_bid, cap = suggest_bid(cpc if not np.isnan(cpc) else 0.5, target_acos, cvr_bayes, asp, min_bid, max_bid, "increase", inc_pct)
                action = "bid_increase"
                suggested_bid = new_bid
                reason.append(f"ROAS≥{round(1/target_acos,2)} & IS<{floor:.2f} (growth) | cap={round(cap,4)}")

            # Efficiency if ACOS too high
            elif acos > 0 and acos > target_acos:
                new_bid, cap = suggest_bid(cpc if not np.isnan(cpc) else 0.5, target_acos, cvr_bayes, asp, min_bid, max_bid, "decrease", dec_pct)
                action = "bid_decrease"
                suggested_bid = new_bid
                reason.append(f"ACOS>{target_acos:.2f} (efficiency) | cap={round(cap,4)}")

            else:
                # Maintain but compute cap-based neutral bid for info
                new_bid, cap = suggest_bid(cpc if not np.isnan(cpc) else 0.5, target_acos, cvr_bayes, asp, min_bid, max_bid, None, None)
                reason.append(f"Maintain; cap={round(cap,4)}")

        recs.append({
            "search_term": term,
            "brand_term": is_brand,
            "impressions": int(r.get("impressions", 0) or 0),
            "clicks": int(clicks),
            "orders": int(orders),
            "spend": round(spend, 4),
            "sales": round(sales, 4),
            "cpc": round(float(cpc), 4) if not (pd.isna(cpc) or cpc is None) else "",
            "ctr": round(r.get("ctr", 0.0), 4) if "ctr" in r else "",
            "cvr_raw": round(safe_div(orders, clicks), 4) if clicks>0 else 0.0,
            "cvr_bayes": round(cvr_bayes, 4),
            "asp": round(asp, 4),
            "acos": round(acos, 4) if acos>0 else 0.0,
            "roas": round(roas, 4) if roas>0 else 0.0,
            "impression_share": round(is_share, 4) if not pd.isna(is_share) else "",
            "action": action,
            "suggested_bid": suggested_bid,
            "reason": "; ".join(reason)
        })

    out_df = pd.DataFrame(recs)
    out_path = args.out_csv
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote recommendations to: {out_path}")
    print(out_df.head(10).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_csv", required=True, help="Path to Sponsored_Product_Search_term_report CSV")
    parser.add_argument("--is_csv", required=True, help="Path to Sponsored_Product_Search_term_impression_share_report CSV")
    parser.add_argument("--out_csv", default="./agent_recommendations.csv", help="Output CSV for recommendations")
    parser.add_argument("--config", default="./config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args)

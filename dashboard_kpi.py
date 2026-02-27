# dashboard_kpi.py
from __future__ import annotations

from dataclasses import dataclass
import json
from turtle import st
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# -----------------------------
# Detection
# -----------------------------
@dataclass
class DetectedSchema:
    date_col: Optional[str]
    revenue_col: Optional[str]
    profit_col: Optional[str]
    cost_col: Optional[str]
    discount_col: Optional[str]
    quantity_col: Optional[str]
    category_cols: List[str]


def _norm(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).strip()


def detect_schema(df: pd.DataFrame) -> DetectedSchema:
    cols = list(df.columns)
    norm = {c: _norm(c) for c in cols}

    # Candidates
    date_keywords = ["date", "order date", "transaction date", "purchase date", "created", "timestamp", "month"]
    revenue_keywords = ["revenue", "sales", "amount", "total", "price", "subtotal", "net sales"]
    profit_keywords = ["profit", "margin"]
    cost_keywords = ["cost", "cogs", "expense", "expenses"]
    discount_keywords = ["discount", "disc"]
    qty_keywords = ["qty", "quantity", "units", "unit sold", "count", "volume"]

    def pick_by_keywords(keywords: List[str]) -> Optional[str]:
        # exact-ish match first
        for c in cols:
            c_norm = norm[c]
            if any(k in c_norm for k in keywords):
                return c
        return None

    date_col = pick_by_keywords(date_keywords)
    # Fallback: find any non-numeric column that parses to a reasonable date range
    if date_col is None:
        for c in cols:
            parsed = _try_parse_dates(df[c])
            if parsed is not None:
                date_col = c
                break
    revenue_col = pick_by_keywords(revenue_keywords)
    profit_col = pick_by_keywords(profit_keywords)
    cost_col = pick_by_keywords(cost_keywords)
    discount_col = pick_by_keywords(discount_keywords)
    quantity_col = pick_by_keywords(qty_keywords)


    # Prefer explicit revenue column names over unit price
    lower_cols = {str(c).strip().lower(): c for c in cols}

    # Strong preference if these exist
    for preferred in ["revenue", "total_revenue", "sales", "total_sales", "net_sales", "net sales"]:
        if preferred in lower_cols:
            revenue_col = lower_cols[preferred]
            break

    # Avoid mistakenly using unit_price if revenue exists
    if revenue_col and str(revenue_col).strip().lower() in ["unit_price", "price", "unit price"]:
        if "revenue" in lower_cols:
            revenue_col = lower_cols["revenue"]

    # Category columns = low-ish cardinality object/category (exclude date)
    category_cols: List[str] = []
    for c in cols:
        if c == date_col:
            continue
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            nunique = s.nunique(dropna=True)
            if 2 <= nunique <= max(25, int(0.05 * len(df))):  # reasonable category range
                category_cols.append(c)

    # If no clear revenue but we have price*qty possibility
    if revenue_col is None and quantity_col is not None:
        # try to find a price/unit column
        price_like = pick_by_keywords(["unit price", "price", "rate"])
        if price_like is not None and pd.api.types.is_numeric_dtype(df[price_like]) and pd.api.types.is_numeric_dtype(df[quantity_col]):
            # create a virtual revenue concept (computed later)
            revenue_col = "__computed_revenue__"
    return DetectedSchema(
        date_col=date_col,
        revenue_col=revenue_col,
        profit_col=profit_col,
        cost_col=cost_col,
        discount_col=discount_col,
        quantity_col=quantity_col,
        category_cols=category_cols,
    )


# -----------------------------
# KPI Computation
# -----------------------------
def _safe_sum(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nan_to_num(x, nan=0.0).sum())


def _safe_mean(x: pd.Series) -> Optional[float]:
    x = pd.to_numeric(x, errors="coerce")
    if x.dropna().empty:
        return None
    return float(x.mean())


def _format_money(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"${v:,.0f}"


def _format_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v*100:,.1f}%"


def _try_parse_dates(s: pd.Series) -> Optional[pd.Series]:
    # Never treat numeric columns as dates (prevents lifetime_revenue -> "date")
    if pd.api.types.is_numeric_dtype(s):
        return None

    try:
        dt = pd.to_datetime(s, errors="coerce")
        valid = dt.notna().sum()
        if valid < max(10, int(0.2 * len(s))):
            return None

        # Sanity check: reject obviously wrong ranges
        years = dt.dropna().dt.year
        if years.empty:
            return None
        if years.min() < 1990 or years.max() > 2100:
            return None

        return dt
    except Exception:
        return None


def compute_dashboard(df: pd.DataFrame, schema: DetectedSchema) -> Dict:
    work = df.copy()

    # Guard against duplicate column labels (prevents pandas ValueError)
    if not work.columns.is_unique:
        cols = [str(c).strip() for c in work.columns]
        seen = {}
        new_cols = []
        for c in cols:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c}.{seen[c]}")
        work.columns = new_cols


    # Guard against duplicate column labels
    if not work.columns.is_unique:
        cols = [str(c).strip() for c in work.columns]
        seen = {}
        new_cols = []
        for c in cols:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c}.{seen[c]}")
        work.columns = new_cols

    # Ensure date
    date_col = schema.date_col
    if date_col is not None:
        parsed = _try_parse_dates(work[date_col])
        if parsed is None:
            date_col = None
        else:
            work[date_col] = parsed

    # Compute revenue if virtual
    revenue_col = schema.revenue_col
    if revenue_col == "__computed_revenue__":
        # find price + qty
        qty = schema.quantity_col
        price = None
        for c in work.columns:
            c_norm = _norm(c)
            if "price" in c_norm and pd.api.types.is_numeric_dtype(work[c]):
                price = c
                break
        if qty and price:
            work["__computed_revenue__"] = pd.to_numeric(work[qty], errors="coerce") * pd.to_numeric(work[price], errors="coerce")
        else:
            revenue_col = None

    # KPIs
    total_revenue = _safe_sum(work[revenue_col]) if revenue_col and revenue_col in work.columns else None
    total_profit = _safe_sum(work[schema.profit_col]) if schema.profit_col and schema.profit_col in work.columns else None

    profit_margin = None
    if total_profit is not None and total_revenue not in (None, 0.0):
        profit_margin = total_profit / total_revenue

    avg_discount = None
    if schema.discount_col and schema.discount_col in work.columns:
        avg_discount = _safe_mean(work[schema.discount_col])
        # if discount looks like 0-100 scale, convert to 0-1
        if avg_discount is not None and avg_discount > 1.5:
            avg_discount = avg_discount / 100.0

    total_qty = _safe_sum(work[schema.quantity_col]) if schema.quantity_col and schema.quantity_col in work.columns else None

    # Trends
    trend = None
    if date_col and revenue_col and revenue_col in work.columns:
        tmp = work[[date_col, revenue_col]].dropna()
        if not tmp.empty:
            tmp = tmp.sort_values(date_col)
            tmp["period"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
            trend = tmp.groupby("period")[revenue_col].sum().reset_index().rename(columns={revenue_col: "revenue"})

    # Profit trend (NEW)
    profit_trend = None
    if date_col and schema.profit_col and schema.profit_col in work.columns:
        tmp = work[[date_col, schema.profit_col]].dropna()
        if not tmp.empty:
            tmp = tmp.sort_values(date_col)
            tmp["period"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
            profit_trend = (
                tmp.groupby("period")[schema.profit_col]
                .sum()
                .reset_index()
                .rename(columns={schema.profit_col: "profit"})
            )

    # Breakdown
    breakdown = None
    best_category_col = schema.category_cols[0] if schema.category_cols else None
    if best_category_col and revenue_col and revenue_col in work.columns:
        tmp = work[[best_category_col, revenue_col]].dropna()
        if not tmp.empty:
            breakdown = (
                tmp.groupby(best_category_col)[revenue_col]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
                .rename(columns={best_category_col: "category", revenue_col: "revenue"})
            )

    # Profit breakdown (NEW)
    profit_breakdown = None
    if best_category_col and schema.profit_col and schema.profit_col in work.columns:
        tmp = work[[best_category_col, schema.profit_col]].dropna()
        if not tmp.empty:
            profit_breakdown = (
                tmp.groupby(best_category_col)[schema.profit_col]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
                .rename(columns={best_category_col: "category", schema.profit_col: "profit"})
            )

    def _mom_change(series: pd.Series) -> Optional[float]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) < 2:
            return None
        prev = float(s.iloc[-2])
        last = float(s.iloc[-1])
        if prev == 0:
            return None
        return (last - prev) / prev

    def _overall_change(series: pd.Series) -> Optional[float]:
        """Overall % change from first to last non-null point."""
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) < 2:
            return None
        first = float(s.iloc[0])
        last = float(s.iloc[-1])
        if first == 0:
            return None
        return (last - first) / first

    # -----------------------------
    # Executive insights (NEW)
    # -----------------------------
    exec_insights = {}

    # MoM changes
    if trend is not None and not trend.empty:
        exec_insights["revenue_mom_pct"] = _mom_change(trend["revenue"])
    else:
        exec_insights["revenue_mom_pct"] = None

    if profit_trend is not None and not profit_trend.empty:
        exec_insights["profit_mom_pct"] = _mom_change(profit_trend["profit"])
    else:
        exec_insights["profit_mom_pct"] = None

    if trend is not None and not trend.empty:
        exec_insights["revenue_overall_pct"] = _overall_change(trend["revenue"])
    else:
        exec_insights["revenue_overall_pct"] = None

    if profit_trend is not None and not profit_trend.empty:
        exec_insights["profit_overall_pct"] = _overall_change(profit_trend["profit"])
    else:
        exec_insights["profit_overall_pct"] = None

    # Best/Worst category by profit
    if profit_breakdown is not None and not profit_breakdown.empty:
        best_row = profit_breakdown.sort_values("profit", ascending=False).iloc[0]
        worst_row = profit_breakdown.sort_values("profit", ascending=True).iloc[0]
        exec_insights["best_category_by_profit"] = {
            "category": str(best_row["category"]),
            "profit": float(best_row["profit"]),
        }
        exec_insights["worst_category_by_profit"] = {
            "category": str(worst_row["category"]),
            "profit": float(worst_row["profit"]),
        }
    else:
        exec_insights["best_category_by_profit"] = None
        exec_insights["worst_category_by_profit"] = None

    # Risk flags
    neg_profit_rows = None
    if schema.profit_col and schema.profit_col in work.columns and pd.api.types.is_numeric_dtype(work[schema.profit_col]):
        neg_profit_rows = int((pd.to_numeric(work[schema.profit_col], errors="coerce") < 0).sum())
    exec_insights["negative_profit_rows"] = neg_profit_rows

    exec_insights["avg_discount"] = avg_discount
    exec_insights["high_discount_flag"] = (avg_discount is not None and avg_discount >= 0.30)

    # A compact dashboard spec (agent-friendly)
    spec = {
        "detected": {
            "date_col": schema.date_col,
            "revenue_col": schema.revenue_col,
            "profit_col": schema.profit_col,
            "cost_col": schema.cost_col,
            "discount_col": schema.discount_col,
            "quantity_col": schema.quantity_col,
            "category_cols": schema.category_cols,
            "used_date_col": date_col,
        },
        "kpis": [
            {"label": "Total Revenue", "value": total_revenue, "display": _format_money(total_revenue)},
            {"label": "Total Profit", "value": total_profit, "display": _format_money(total_profit)},
            {"label": "Profit Margin", "value": profit_margin, "display": _format_pct(profit_margin)},
            {"label": "Avg Discount", "value": avg_discount, "display": _format_pct(avg_discount)},
            {"label": "Total Quantity", "value": total_qty, "display": f"{total_qty:,.0f}" if total_qty is not None else "—"},
        ],
        "trend": trend,
        "profit_trend": profit_trend,                 # NEW
        "breakdown": breakdown,
        "profit_breakdown": profit_breakdown,
        "exec_insights": exec_insights,
    }


    return spec


# -----------------------------
# Executive Summary (LLM-friendly prompt builder)
# -----------------------------
def build_exec_summary_prompt(spec: Dict) -> str:
    kpis = {k["label"]: k["display"] for k in spec.get("kpis", [])}
    detected = spec.get("detected", {})
    ins = spec.get("exec_insights", {})
    has_trend = spec.get("trend") is not None
    has_breakdown = spec.get("breakdown") is not None

    def fmt_pct(x):
        return "—" if x is None else f"{x*100:.1f}%"

    best = ins.get("best_category_by_profit")
    worst = ins.get("worst_category_by_profit")

    best_txt = "—" if not best else f"{best['category']} (${best['profit']:,.0f})"
    worst_txt = "—" if not worst else f"{worst['category']} (${worst['profit']:,.0f})"

    prompt = f"""
You are an executive data analyst. Write a short executive summary (3-5 bullet points).
Be specific, avoid fluff, and use the KPI values provided.

KPI values:
- Total Revenue: {kpis.get("Total Revenue","—")}
- Total Profit: {kpis.get("Total Profit","—")}
- Profit Margin: {kpis.get("Profit Margin","—")}
- Avg Discount: {kpis.get("Avg Discount","—")}
- Total Quantity: {kpis.get("Total Quantity","—")}

- Revenue MoM change: {fmt_pct(ins.get("revenue_mom_pct"))}
- Profit MoM change: {fmt_pct(ins.get("profit_mom_pct"))}
- Revenue overall change (first→last): {fmt_pct(ins.get("revenue_overall_pct"))}
- Profit overall change (first→last): {fmt_pct(ins.get("profit_overall_pct"))}
- Best category by profit: {best_txt}
- Worst category by profit: {worst_txt}
- Negative profit rows: {ins.get("negative_profit_rows")}
- High discount flag: {ins.get("high_discount_flag")}

Detected columns:
{detected}

Charts available:
- Trend chart available: {has_trend}
- Category breakdown available: {has_breakdown}

Rules:
- If trend chart is available, mention whether revenue is rising/falling/flat based on the latest periods.
- If category breakdown is available, mention the top category and its contribution.
- If information is missing, say "Insufficient data to determine X" instead of guessing.
- If MoM is None, don’t claim growth/decline.
"""
    return prompt.strip()


# --- sanity exports ---
# Make sure these exist at top-level:
# detect_schema, compute_dashboard, build_exec_summary_prompt
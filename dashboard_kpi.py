# dashboard_kpi.py  — fully rewritten
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
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
    numeric_cols: List[str]
    categorical_cols: List[str]
    target_col: Optional[str]
    dashboard_type: str   # "sales" | "classification" | "generic"


def _norm(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).strip()


def _try_parse_dates(s: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_numeric_dtype(s):
        return None
    try:
        dt = pd.to_datetime(s, errors="coerce")
        valid = int(dt.notna().sum())
        if valid < max(10, int(0.2 * len(s))):
            return None
        years = dt.dropna().dt.year
        if years.empty or years.min() < 1990 or years.max() > 2100:
            return None
        return dt
    except Exception:
        return None


def _pick_by_keywords(cols: List[str], norm: Dict[str, str], keywords: List[str]) -> Optional[str]:
    for c in cols:
        if any(k in norm[c] for k in keywords):
            return c
    return None


def _make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    df = df.copy()
    cols = [str(c).strip() for c in df.columns]
    seen: Dict[str, int] = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}.{seen[c]}")
    df.columns = new_cols
    return df


def _detect_target_col(df: pd.DataFrame) -> Optional[str]:
    target_keywords = ["target", "label", "class", "outcome", "diagnosis", "y",
                       "disease", "malignant", "benign", "survival", "status", "response"]
    cols = list(df.columns)
    norm = {c: _norm(c) for c in cols}
    cand = _pick_by_keywords(cols, norm, target_keywords)
    if cand is not None and 2 <= int(df[cand].nunique(dropna=True)) <= 50:
        return cand
    best, best_score = None, 0
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        nunique = int(s.nunique(dropna=True))
        if 2 <= nunique <= 10:
            score = 100 - nunique
            if score > best_score:
                best, best_score = c, score
    return best


def detect_schema(df: pd.DataFrame) -> DetectedSchema:
    df = _make_unique_columns(df)
    cols = list(df.columns)
    norm = {c: _norm(c) for c in cols}

    date_col = _pick_by_keywords(cols, norm, ["date", "order date", "transaction date",
                                               "purchase date", "created", "timestamp", "month"])
    if date_col is None:
        for c in cols:
            if _try_parse_dates(df[c]) is not None:
                date_col = c
                break

    revenue_col    = _pick_by_keywords(cols, norm, ["revenue", "sales", "amount", "total", "price", "subtotal", "net sales"])
    profit_col     = _pick_by_keywords(cols, norm, ["profit", "margin"])
    cost_col       = _pick_by_keywords(cols, norm, ["cost", "cogs", "expense", "expenses"])
    discount_col   = _pick_by_keywords(cols, norm, ["discount", "disc"])
    quantity_col   = _pick_by_keywords(cols, norm, ["qty", "quantity", "units", "unit sold", "count", "volume"])

    lower_cols = {str(c).strip().lower(): c for c in cols}
    for preferred in ["revenue", "total_revenue", "sales", "total_sales", "net_sales", "net sales"]:
        if preferred in lower_cols:
            revenue_col = lower_cols[preferred]
            break

    category_cols: List[str] = []
    for c in cols:
        if c == date_col:
            continue
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            nunique = int(s.nunique(dropna=True))
            if 2 <= nunique <= max(25, int(0.05 * len(df))):
                category_cols.append(c)

    if revenue_col is None and quantity_col is not None:
        price_like = _pick_by_keywords(cols, norm, ["unit price", "price", "rate"])
        if price_like and pd.api.types.is_numeric_dtype(df[price_like]) and pd.api.types.is_numeric_dtype(df[quantity_col]):
            revenue_col = "__computed_revenue__"

    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in cols if c not in numeric_cols and c != date_col]
    target_col = _detect_target_col(df)

    if (revenue_col and revenue_col != "__computed_revenue__") or profit_col or revenue_col == "__computed_revenue__":
        dashboard_type = "sales"
    elif target_col is not None and 2 <= int(df[target_col].nunique(dropna=True)) <= 20 and len(numeric_cols) >= 2:
        dashboard_type = "classification"
    else:
        dashboard_type = "generic"

    return DetectedSchema(
        date_col=date_col, revenue_col=revenue_col, profit_col=profit_col,
        cost_col=cost_col, discount_col=discount_col, quantity_col=quantity_col,
        category_cols=category_cols, numeric_cols=numeric_cols,
        categorical_cols=categorical_cols, target_col=target_col,
        dashboard_type=dashboard_type,
    )


# -----------------------------
# KPI helpers
# -----------------------------
def _safe_sum(x: pd.Series) -> float:
    return float(np.nan_to_num(pd.to_numeric(x, errors="coerce").values, nan=0.0).sum())

def _safe_mean(x: pd.Series) -> Optional[float]:
    v = pd.to_numeric(x, errors="coerce").dropna()
    return float(v.mean()) if not v.empty else None

def _fmt_money(v): return "—" if v is None else f"${v:,.0f}"
def _fmt_int(v):   return "—" if v is None else f"{v:,.0f}"
def _fmt_pct(v):   return "—" if v is None else f"{v*100:,.1f}%"


def _top_correlations(df: pd.DataFrame, cols: List[str], top_k: int = 12) -> pd.DataFrame:
    if len(cols) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr"])
    x = df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    cols2 = x.columns.tolist()
    if len(cols2) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr"])
    corr = x.corr(numeric_only=True)
    pairs = [(cols2[i], cols2[j], float(corr.loc[cols2[i], cols2[j]]))
             for i in range(len(cols2)) for j in range(i+1, len(cols2))
             if not pd.isna(corr.loc[cols2[i], cols2[j]])]
    if not pairs:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr"])
    dfp = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "corr"])
    dfp["abs_corr"] = dfp["corr"].abs()
    return dfp.sort_values("abs_corr", ascending=False).head(top_k).drop(columns=["abs_corr"])


# -----------------------------
# Dashboard spec builder (data only, no Streamlit)
# -----------------------------
def compute_dashboard(df: pd.DataFrame, schema: DetectedSchema) -> Dict:
    work = _make_unique_columns(df.copy())
    used_date_col = schema.date_col
    if used_date_col:
        parsed = _try_parse_dates(work[used_date_col])
        if parsed is None:
            used_date_col = None
        else:
            work[used_date_col] = parsed

    rows, cols_n = work.shape
    missing_cells = int(work.isna().sum().sum())
    dup_rows = int(work.duplicated().sum())

    if schema.dashboard_type == "sales":
        return _build_sales_spec(work, schema, used_date_col, rows, missing_cells, dup_rows)
    elif schema.dashboard_type == "classification":
        return _build_classification_spec(work, schema, rows, cols_n, missing_cells, dup_rows)
    else:
        return _build_generic_spec(work, schema, rows, cols_n, missing_cells, dup_rows)


def _build_sales_spec(work, schema, used_date_col, rows, missing_cells, dup_rows):
    revenue_col = schema.revenue_col
    if revenue_col == "__computed_revenue__":
        qty, price = schema.quantity_col, None
        for c in work.columns:
            if "price" in _norm(c) and pd.api.types.is_numeric_dtype(work[c]):
                price = c; break
        if qty and price:
            work["__computed_revenue__"] = pd.to_numeric(work[qty], errors="coerce") * pd.to_numeric(work[price], errors="coerce")
        else:
            revenue_col = None

    total_revenue = _safe_sum(work[revenue_col]) if revenue_col and revenue_col in work.columns else None
    total_profit  = _safe_sum(work[schema.profit_col]) if schema.profit_col and schema.profit_col in work.columns else None
    profit_margin = (total_profit / total_revenue) if total_profit and total_revenue else None
    avg_discount  = None
    if schema.discount_col and schema.discount_col in work.columns:
        avg_discount = _safe_mean(work[schema.discount_col])
        if avg_discount and avg_discount > 1.5:
            avg_discount /= 100.0
    total_qty = _safe_sum(work[schema.quantity_col]) if schema.quantity_col and schema.quantity_col in work.columns else None

    trend = profit_trend = breakdown = profit_breakdown = None
    if used_date_col and revenue_col and revenue_col in work.columns:
        tmp = work[[used_date_col, revenue_col]].dropna().sort_values(used_date_col)
        tmp = tmp.copy()
        tmp["period"] = tmp[used_date_col].dt.to_period("M").dt.to_timestamp()
        trend = tmp.groupby("period")[revenue_col].sum().reset_index().rename(columns={revenue_col: "revenue"})
    if used_date_col and schema.profit_col and schema.profit_col in work.columns:
        tmp = work[[used_date_col, schema.profit_col]].dropna().sort_values(used_date_col)
        tmp = tmp.copy()
        tmp["period"] = tmp[used_date_col].dt.to_period("M").dt.to_timestamp()
        profit_trend = tmp.groupby("period")[schema.profit_col].sum().reset_index().rename(columns={schema.profit_col: "profit"})

    best_cat = schema.category_cols[0] if schema.category_cols else None
    if best_cat and revenue_col and revenue_col in work.columns:
        tmp = work[[best_cat, revenue_col]].dropna()
        breakdown = tmp.groupby(best_cat)[revenue_col].sum().sort_values(ascending=False).head(15).reset_index().rename(
            columns={best_cat: "category", revenue_col: "revenue"})
    if best_cat and schema.profit_col and schema.profit_col in work.columns:
        tmp = work[[best_cat, schema.profit_col]].dropna()
        profit_breakdown = tmp.groupby(best_cat)[schema.profit_col].sum().sort_values(ascending=False).head(15).reset_index().rename(
            columns={best_cat: "category", schema.profit_col: "profit"})

    def _mom(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) < 2: return None
        p, l = float(s.iloc[-2]), float(s.iloc[-1])
        return None if p == 0 else (l - p) / p

    def _overall(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) < 2: return None
        f, l = float(s.iloc[0]), float(s.iloc[-1])
        return None if f == 0 else (l - f) / f

    neg_profit_rows = None
    if schema.profit_col and schema.profit_col in work.columns and pd.api.types.is_numeric_dtype(work[schema.profit_col]):
        neg_profit_rows = int((pd.to_numeric(work[schema.profit_col], errors="coerce") < 0).sum())

    best_cat_profit = worst_cat_profit = None
    if isinstance(profit_breakdown, pd.DataFrame) and not profit_breakdown.empty:
        best_cat_profit  = {"category": str(profit_breakdown.iloc[0]["category"]),  "profit": float(profit_breakdown.iloc[0]["profit"])}
        worst_cat_profit = {"category": str(profit_breakdown.iloc[-1]["category"]), "profit": float(profit_breakdown.iloc[-1]["profit"])}

    exec_insights = {
        "revenue_mom_pct":       _mom(trend["revenue"])          if isinstance(trend, pd.DataFrame) and not trend.empty else None,
        "profit_mom_pct":        _mom(profit_trend["profit"])     if isinstance(profit_trend, pd.DataFrame) and not profit_trend.empty else None,
        "revenue_overall_pct":   _overall(trend["revenue"])       if isinstance(trend, pd.DataFrame) and not trend.empty else None,
        "profit_overall_pct":    _overall(profit_trend["profit"]) if isinstance(profit_trend, pd.DataFrame) and not profit_trend.empty else None,
        "avg_discount": avg_discount,
        "high_discount_flag": (avg_discount is not None and avg_discount >= 0.30),
        "negative_profit_rows": neg_profit_rows,
        "best_category_by_profit":  best_cat_profit,
        "worst_category_by_profit": worst_cat_profit,
    }

    # drill-down data: per category per period (for interactive Plotly)
    drill_data = None
    if used_date_col and best_cat and revenue_col and revenue_col in work.columns:
        tmp = work[[used_date_col, best_cat, revenue_col]].dropna().copy()
        tmp["period"] = tmp[used_date_col].dt.to_period("M").dt.to_timestamp()
        drill_data = tmp.groupby(["period", best_cat])[revenue_col].sum().reset_index().rename(
            columns={best_cat: "category", revenue_col: "revenue"})

    return {
        "dashboard_type": "sales",
        "detected": {
            "date_col": schema.date_col, "revenue_col": schema.revenue_col,
            "profit_col": schema.profit_col, "cost_col": schema.cost_col,
            "discount_col": schema.discount_col, "quantity_col": schema.quantity_col,
            "category_cols": schema.category_cols, "used_date_col": used_date_col,
        },
        "kpis": [
            {"label": "Rows",           "value": rows,          "display": _fmt_int(rows)},
            {"label": "Missing Cells",  "value": missing_cells, "display": _fmt_int(missing_cells)},
            {"label": "Total Revenue",  "value": total_revenue, "display": _fmt_money(total_revenue)},
            {"label": "Total Profit",   "value": total_profit,  "display": _fmt_money(total_profit)},
            {"label": "Profit Margin",  "value": profit_margin, "display": _fmt_pct(profit_margin)},
            {"label": "Avg Discount",   "value": avg_discount,  "display": _fmt_pct(avg_discount)},
            {"label": "Total Quantity", "value": total_qty,     "display": _fmt_int(total_qty)},
        ],
        "trend": trend, "profit_trend": profit_trend,
        "breakdown": breakdown, "profit_breakdown": profit_breakdown,
        "drill_data": drill_data,
        "exec_insights": exec_insights,
        "raw": work,
        "schema": schema,
    }


def _build_classification_spec(work, schema, rows, cols_n, missing_cells, dup_rows):
    target = schema.target_col
    y = work[target] if target and target in work.columns else None
    class_balance = None
    if y is not None:
        vc = y.astype("object").value_counts(dropna=False)
        class_balance = vc.reset_index()
        class_balance.columns = ["class", "count"]

    num_cols = [c for c in schema.numeric_cols if c in work.columns]
    sep_table = None; top_features = []
    if y is not None and len(num_cols) >= 2:
        classes = y.dropna().astype("object").unique().tolist()
        if 2 <= len(classes) <= 10:
            tmp = work[[target] + num_cols].copy()
            tmp[target] = tmp[target].astype("object")
            means = tmp.groupby(target)[num_cols].mean(numeric_only=True)
            if len(means.index) >= 2:
                if len(means.index) == 2:
                    a, b = means.index[0], means.index[1]
                    diff = (means.loc[a] - means.loc[b]).abs().sort_values(ascending=False)
                    top_features = diff.head(8).index.tolist()
                    sep_table = pd.DataFrame({"feature": diff.head(12).index, "abs_mean_diff": diff.head(12).values})
                else:
                    spread = means.std(axis=0).sort_values(ascending=False)
                    top_features = spread.head(8).index.tolist()
                    sep_table = pd.DataFrame({"feature": spread.head(12).index, "class_mean_std": spread.head(12).values})

    corr_pairs = _top_correlations(work, num_cols[:40])
    numeric_summary = None
    if num_cols:
        desc = work[num_cols[:20]].describe().T
        desc = desc.rename(columns={"50%": "median"}).reset_index().rename(columns={"index": "feature"})
        keep = ["feature", "mean", "std", "min", "median", "max"]
        numeric_summary = desc[[c for c in keep if c in desc.columns]]

    return {
        "dashboard_type": "classification",
        "detected": {"target_col": target, "numeric_cols_count": len(num_cols), "categorical_cols_count": len(schema.categorical_cols)},
        "kpis": [
            {"label": "Rows",             "value": rows,          "display": _fmt_int(rows)},
            {"label": "Columns",          "value": cols_n,        "display": _fmt_int(cols_n)},
            {"label": "Missing Cells",    "value": missing_cells, "display": _fmt_int(missing_cells)},
            {"label": "Duplicate Rows",   "value": dup_rows,      "display": _fmt_int(dup_rows)},
            {"label": "Numeric Features", "value": len(num_cols), "display": _fmt_int(len(num_cols))},
            {"label": "Target",           "value": target,        "display": str(target) if target else "—"},
        ],
        "class_balance": class_balance, "top_separation": sep_table,
        "top_features": top_features,   "corr_pairs": corr_pairs,
        "numeric_summary": numeric_summary,
        "raw": work, "schema": schema,
    }


def _build_generic_spec(work, schema, rows, cols_n, missing_cells, dup_rows):
    num_cols = [c for c in schema.numeric_cols if c in work.columns]
    corr_pairs = _top_correlations(work, num_cols[:40])
    cat_summaries = []
    for c in schema.categorical_cols[:3]:
        if c in work.columns:
            vc = work[c].astype("object").value_counts(dropna=False).head(10).reset_index()
            vc.columns = [c, "count"]
            cat_summaries.append({"col": c, "top_counts": vc})
    numeric_summary = None
    if num_cols:
        desc = work[num_cols[:20]].describe().T
        desc = desc.rename(columns={"50%": "median"}).reset_index().rename(columns={"index": "feature"})
        keep = ["feature", "mean", "std", "min", "median", "max"]
        numeric_summary = desc[[c for c in keep if c in desc.columns]]

    return {
        "dashboard_type": "generic",
        "detected": {"numeric_cols_count": len(num_cols), "categorical_cols_count": len(schema.categorical_cols), "date_col": schema.date_col},
        "kpis": [
            {"label": "Rows",                "value": rows,                          "display": _fmt_int(rows)},
            {"label": "Columns",             "value": cols_n,                        "display": _fmt_int(cols_n)},
            {"label": "Missing Cells",       "value": missing_cells,                 "display": _fmt_int(missing_cells)},
            {"label": "Duplicate Rows",      "value": dup_rows,                      "display": _fmt_int(dup_rows)},
            {"label": "Numeric Columns",     "value": len(num_cols),                 "display": _fmt_int(len(num_cols))},
            {"label": "Categorical Columns", "value": len(schema.categorical_cols),  "display": _fmt_int(len(schema.categorical_cols))},
        ],
        "corr_pairs": corr_pairs, "numeric_summary": numeric_summary,
        "categorical_summaries": cat_summaries,
        "raw": work, "schema": schema,
    }


# -----------------------------
# Executive summary prompt
# -----------------------------
def build_exec_summary_prompt(spec: Dict) -> str:
    dtype = spec.get("dashboard_type")
    if dtype != "sales":
        return (f"You are an analyst. Write 3-5 concise bullet points summarizing the dashboard.\n"
                f"Dashboard type: {dtype}\n")
    kpis = {k["label"]: k["display"] for k in spec.get("kpis", [])}
    ins  = spec.get("exec_insights", {})
    det  = spec.get("detected", {})
    def fp(x): return "—" if x is None else f"{x*100:.1f}%"
    best  = ins.get("best_category_by_profit")
    worst = ins.get("worst_category_by_profit")
    return f"""
You are an executive data analyst. Write a short executive summary (3-5 bullet points).
Be specific, avoid fluff, use the KPI values.

KPIs:
- Total Revenue: {kpis.get("Total Revenue","—")}
- Total Profit: {kpis.get("Total Profit","—")}
- Profit Margin: {kpis.get("Profit Margin","—")}
- Avg Discount: {kpis.get("Avg Discount","—")}
- Total Quantity: {kpis.get("Total Quantity","—")}
- Revenue MoM: {fp(ins.get("revenue_mom_pct"))}
- Profit MoM: {fp(ins.get("profit_mom_pct"))}
- Revenue overall: {fp(ins.get("revenue_overall_pct"))}
- Profit overall: {fp(ins.get("profit_overall_pct"))}
- Best category by profit: {"—" if not best else f"{best['category']} (${best['profit']:,.0f})"}
- Worst category by profit: {"—" if not worst else f"{worst['category']} (${worst['profit']:,.0f})"}
- Negative profit rows: {ins.get("negative_profit_rows")}
- High discount flag: {ins.get("high_discount_flag")}

Detected columns: {det}
Charts available: trend={spec.get('trend') is not None}, breakdown={spec.get('breakdown') is not None}

Rules:
- If MoM is None do not claim growth/decline.
- Say "Insufficient data" for any missing metric.
""".strip()
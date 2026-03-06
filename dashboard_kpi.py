# dashboard_kpi.py
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
    # Sales/business oriented (kept for backward compatibility)
    date_col: Optional[str]
    revenue_col: Optional[str]
    profit_col: Optional[str]
    cost_col: Optional[str]
    discount_col: Optional[str]
    quantity_col: Optional[str]
    category_cols: List[str]

    # Universal detection
    numeric_cols: List[str]
    categorical_cols: List[str]
    target_col: Optional[str]          # e.g., diagnosis/label/target/class
    dashboard_type: str                # "sales" | "classification" | "generic"


def _norm(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).strip()


def _try_parse_dates(s: pd.Series) -> Optional[pd.Series]:
    # Never treat numeric columns as dates
    if pd.api.types.is_numeric_dtype(s):
        return None

    try:
        dt = pd.to_datetime(s, errors="coerce")
        valid = int(dt.notna().sum())
        if valid < max(10, int(0.2 * len(s))):
            return None

        years = dt.dropna().dt.year
        if years.empty:
            return None

        # sanity range
        if years.min() < 1990 or years.max() > 2100:
            return None

        return dt
    except Exception:
        return None


def _pick_by_keywords(cols: List[str], norm: Dict[str, str], keywords: List[str]) -> Optional[str]:
    for c in cols:
        c_norm = norm[c]
        if any(k in c_norm for k in keywords):
            return c
    return None


def _make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.is_unique:
        return df

    df = df.copy()
    cols = [str(c).strip() for c in df.columns]
    seen = {}
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
    # strong keyword candidates first
    target_keywords = [
        "target", "label", "class", "outcome", "diagnosis", "y", "disease",
        "malignant", "benign", "survival", "status", "response"
    ]
    cols = list(df.columns)
    norm = {c: _norm(c) for c in cols}

    cand = _pick_by_keywords(cols, norm, target_keywords)
    if cand is not None:
        # must be low/medium cardinality to behave like a label
        nunique = int(df[cand].nunique(dropna=True))
        if 2 <= nunique <= 50:
            return cand

    # fallback: pick a non-numeric column with low cardinality
    best = None
    best_score = 0
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        nunique = int(s.nunique(dropna=True))
        if nunique < 2:
            continue
        # prefer smaller cardinality (like binary/multiclass label)
        if nunique <= 10:
            score = 100 - nunique
            if score > best_score:
                best = c
                best_score = score
    return best


def detect_schema(df: pd.DataFrame) -> DetectedSchema:
    df = _make_unique_columns(df)
    cols = list(df.columns)
    norm = {c: _norm(c) for c in cols}

    # --- Sales candidates (existing logic) ---
    date_keywords = ["date", "order date", "transaction date", "purchase date", "created", "timestamp", "month"]
    revenue_keywords = ["revenue", "sales", "amount", "total", "price", "subtotal", "net sales"]
    profit_keywords = ["profit", "margin"]
    cost_keywords = ["cost", "cogs", "expense", "expenses"]
    discount_keywords = ["discount", "disc"]
    qty_keywords = ["qty", "quantity", "units", "unit sold", "count", "volume"]

    date_col = _pick_by_keywords(cols, norm, date_keywords)
    if date_col is None:
        for c in cols:
            parsed = _try_parse_dates(df[c])
            if parsed is not None:
                date_col = c
                break

    revenue_col = _pick_by_keywords(cols, norm, revenue_keywords)
    profit_col = _pick_by_keywords(cols, norm, profit_keywords)
    cost_col = _pick_by_keywords(cols, norm, cost_keywords)
    discount_col = _pick_by_keywords(cols, norm, discount_keywords)
    quantity_col = _pick_by_keywords(cols, norm, qty_keywords)

    # Prefer explicit revenue column names over unit price
    lower_cols = {str(c).strip().lower(): c for c in cols}
    for preferred in ["revenue", "total_revenue", "sales", "total_sales", "net_sales", "net sales"]:
        if preferred in lower_cols:
            revenue_col = lower_cols[preferred]
            break

    # Category columns (for sales + also useful in general)
    category_cols: List[str] = []
    for c in cols:
        if c == date_col:
            continue
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            nunique = int(s.nunique(dropna=True))
            if 2 <= nunique <= max(25, int(0.05 * len(df))):
                category_cols.append(c)

    # If no clear revenue but we have price*qty possibility
    if revenue_col is None and quantity_col is not None:
        price_like = _pick_by_keywords(cols, norm, ["unit price", "price", "rate"])
        if price_like is not None and pd.api.types.is_numeric_dtype(df[price_like]) and pd.api.types.is_numeric_dtype(df[quantity_col]):
            revenue_col = "__computed_revenue__"

    # --- Universal columns ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in cols if c not in numeric_cols and c != date_col]

    target_col = _detect_target_col(df)

    # Choose dashboard type
    # Sales if we have revenue or profit
    if (revenue_col and revenue_col != "__computed_revenue__") or profit_col or revenue_col == "__computed_revenue__":
        dashboard_type = "sales"
    else:
        # classification if we have a target_col with small cardinality and enough numeric cols
        if target_col is not None:
            nunique = int(df[target_col].nunique(dropna=True))
            if 2 <= nunique <= 20 and len(numeric_cols) >= 2:
                dashboard_type = "classification"
            else:
                dashboard_type = "generic"
        else:
            dashboard_type = "generic"

    return DetectedSchema(
        date_col=date_col,
        revenue_col=revenue_col,
        profit_col=profit_col,
        cost_col=cost_col,
        discount_col=discount_col,
        quantity_col=quantity_col,
        category_cols=category_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col=target_col,
        dashboard_type=dashboard_type,
    )


# -----------------------------
# KPI Computation helpers
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


def _format_int(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:,.0f}"


def _format_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v*100:,.1f}%"


def _top_categorical_counts(df: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    s = df[col].astype("object")
    vc = s.value_counts(dropna=False).head(n)
    out = vc.reset_index()
    out.columns = [col, "count"]
    return out


def _top_correlations(df: pd.DataFrame, cols: List[str], top_k: int = 12) -> pd.DataFrame:
    if len(cols) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr"])

    x = df[cols].copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.dropna(axis=1, how="all")
    cols2 = x.columns.tolist()
    if len(cols2) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr"])

    corr = x.corr(numeric_only=True)
    # flatten upper triangle
    pairs = []
    for i in range(len(cols2)):
        for j in range(i + 1, len(cols2)):
            a, b = cols2[i], cols2[j]
            v = corr.loc[a, b]
            if pd.isna(v):
                continue
            pairs.append((a, b, float(v)))

    if not pairs:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr"])

    dfp = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "corr"])
    dfp["abs_corr"] = dfp["corr"].abs()
    dfp = dfp.sort_values("abs_corr", ascending=False).head(top_k).drop(columns=["abs_corr"])
    return dfp


# -----------------------------
# Dashboard builder
# -----------------------------
def compute_dashboard(df: pd.DataFrame, schema: DetectedSchema) -> Dict:
    work = _make_unique_columns(df.copy())

    # normalize date column if present
    used_date_col = schema.date_col
    if used_date_col is not None:
        parsed = _try_parse_dates(work[used_date_col])
        if parsed is None:
            used_date_col = None
        else:
            work[used_date_col] = parsed

    # Common KPIs (always available)
    rows, cols = work.shape
    missing_cells = int(work.isna().sum().sum())
    dup_rows = int(work.duplicated().sum())
    num_count = len(schema.numeric_cols)
    cat_count = len(schema.categorical_cols)

    # ---- SALES MODE ----
    if schema.dashboard_type == "sales":
        revenue_col = schema.revenue_col

        # compute revenue if virtual
        if revenue_col == "__computed_revenue__":
            qty = schema.quantity_col
            price = None
            for c in work.columns:
                if "price" in _norm(c) and pd.api.types.is_numeric_dtype(work[c]):
                    price = c
                    break
            if qty and price:
                work["__computed_revenue__"] = pd.to_numeric(work[qty], errors="coerce") * pd.to_numeric(work[price], errors="coerce")
            else:
                revenue_col = None

        total_revenue = _safe_sum(work[revenue_col]) if revenue_col and revenue_col in work.columns else None
        total_profit = _safe_sum(work[schema.profit_col]) if schema.profit_col and schema.profit_col in work.columns else None

        profit_margin = None
        if total_profit is not None and total_revenue not in (None, 0.0):
            profit_margin = total_profit / total_revenue

        avg_discount = None
        if schema.discount_col and schema.discount_col in work.columns:
            avg_discount = _safe_mean(work[schema.discount_col])
            if avg_discount is not None and avg_discount > 1.5:
                avg_discount = avg_discount / 100.0

        total_qty = _safe_sum(work[schema.quantity_col]) if schema.quantity_col and schema.quantity_col in work.columns else None

        # trends
        trend = None
        if used_date_col and revenue_col and revenue_col in work.columns:
            tmp = work[[used_date_col, revenue_col]].dropna()
            if not tmp.empty:
                tmp = tmp.sort_values(used_date_col)
                tmp["period"] = tmp[used_date_col].dt.to_period("M").dt.to_timestamp()
                trend = tmp.groupby("period")[revenue_col].sum().reset_index().rename(columns={revenue_col: "revenue"})

        profit_trend = None
        if used_date_col and schema.profit_col and schema.profit_col in work.columns:
            tmp = work[[used_date_col, schema.profit_col]].dropna()
            if not tmp.empty:
                tmp = tmp.sort_values(used_date_col)
                tmp["period"] = tmp[used_date_col].dt.to_period("M").dt.to_timestamp()
                profit_trend = tmp.groupby("period")[schema.profit_col].sum().reset_index().rename(columns={schema.profit_col: "profit"})

        breakdown = None
        profit_breakdown = None
        best_category_col = schema.category_cols[0] if schema.category_cols else None
        if best_category_col and revenue_col and revenue_col in work.columns:
            tmp = work[[best_category_col, revenue_col]].dropna()
            if not tmp.empty:
                breakdown = (
                    tmp.groupby(best_category_col)[revenue_col]
                    .sum().sort_values(ascending=False).head(10)
                    .reset_index().rename(columns={best_category_col: "category", revenue_col: "revenue"})
                )

        if best_category_col and schema.profit_col and schema.profit_col in work.columns:
            tmp = work[[best_category_col, schema.profit_col]].dropna()
            if not tmp.empty:
                profit_breakdown = (
                    tmp.groupby(best_category_col)[schema.profit_col]
                    .sum().sort_values(ascending=False).head(10)
                    .reset_index().rename(columns={best_category_col: "category", schema.profit_col: "profit"})
                )

        # executive insight helpers
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
            s = pd.to_numeric(series, errors="coerce").dropna()
            if len(s) < 2:
                return None
            first = float(s.iloc[0])
            last = float(s.iloc[-1])
            if first == 0:
                return None
            return (last - first) / first

        exec_insights = {
            "revenue_mom_pct": _mom_change(trend["revenue"]) if isinstance(trend, pd.DataFrame) and not trend.empty else None,
            "profit_mom_pct": _mom_change(profit_trend["profit"]) if isinstance(profit_trend, pd.DataFrame) and not profit_trend.empty else None,
            "revenue_overall_pct": _overall_change(trend["revenue"]) if isinstance(trend, pd.DataFrame) and not trend.empty else None,
            "profit_overall_pct": _overall_change(profit_trend["profit"]) if isinstance(profit_trend, pd.DataFrame) and not profit_trend.empty else None,
            "avg_discount": avg_discount,
            "high_discount_flag": (avg_discount is not None and avg_discount >= 0.30),
        }

        neg_profit_rows = None
        if schema.profit_col and schema.profit_col in work.columns and pd.api.types.is_numeric_dtype(work[schema.profit_col]):
            neg_profit_rows = int((pd.to_numeric(work[schema.profit_col], errors="coerce") < 0).sum())
        exec_insights["negative_profit_rows"] = neg_profit_rows

        # best/worst category by profit
        if isinstance(profit_breakdown, pd.DataFrame) and not profit_breakdown.empty:
            best_row = profit_breakdown.sort_values("profit", ascending=False).iloc[0]
            worst_row = profit_breakdown.sort_values("profit", ascending=True).iloc[0]
            exec_insights["best_category_by_profit"] = {"category": str(best_row["category"]), "profit": float(best_row["profit"])}
            exec_insights["worst_category_by_profit"] = {"category": str(worst_row["category"]), "profit": float(worst_row["profit"])}
        else:
            exec_insights["best_category_by_profit"] = None
            exec_insights["worst_category_by_profit"] = None

        return {
            "dashboard_type": "sales",
            "detected": {
                "date_col": schema.date_col,
                "revenue_col": schema.revenue_col,
                "profit_col": schema.profit_col,
                "cost_col": schema.cost_col,
                "discount_col": schema.discount_col,
                "quantity_col": schema.quantity_col,
                "category_cols": schema.category_cols,
                "used_date_col": used_date_col,
            },
            "kpis": [
                {"label": "Rows", "value": rows, "display": _format_int(rows)},
                {"label": "Missing Cells", "value": missing_cells, "display": _format_int(missing_cells)},
                {"label": "Total Revenue", "value": total_revenue, "display": _format_money(total_revenue)},
                {"label": "Total Profit", "value": total_profit, "display": _format_money(total_profit)},
                {"label": "Profit Margin", "value": profit_margin, "display": _format_pct(profit_margin)},
                {"label": "Avg Discount", "value": avg_discount, "display": _format_pct(avg_discount)},
                {"label": "Total Quantity", "value": total_qty, "display": _format_int(total_qty)},
            ],
            "trend": trend,
            "profit_trend": profit_trend,
            "breakdown": breakdown,
            "profit_breakdown": profit_breakdown,
            "exec_insights": exec_insights,
        }

    # ---- CLASSIFICATION MODE ----
    if schema.dashboard_type == "classification":
        target = schema.target_col
        y = work[target] if target in work.columns else None

        # class balance
        class_balance = None
        if y is not None:
            vc = y.astype("object").value_counts(dropna=False)
            class_balance = vc.reset_index()
            class_balance.columns = ["class", "count"]

        # pick top separating numeric features by mean difference between classes (binary preferred)
        top_features = []
        sep_table = None
        num_cols = [c for c in schema.numeric_cols if c in work.columns]
        if y is not None and len(num_cols) >= 2:
            y_vals = y.dropna().astype("object")
            classes = y_vals.unique().tolist()
            if 2 <= len(classes) <= 10:
                # compute per-class means
                tmp = work[[target] + num_cols].copy()
                tmp[target] = tmp[target].astype("object")
                means = tmp.groupby(target)[num_cols].mean(numeric_only=True)

                if len(means.index) >= 2:
                    # binary: abs(mean1 - mean2)
                    if len(means.index) == 2:
                        a, b = means.index[0], means.index[1]
                        diff = (means.loc[a] - means.loc[b]).abs().sort_values(ascending=False)
                        top_features = diff.head(8).index.tolist()

                        sep_table = pd.DataFrame({
                            "feature": diff.head(12).index,
                            "abs_mean_diff": diff.head(12).values
                        })
                    else:
                        # multiclass: use std of class means
                        spread = means.std(axis=0).sort_values(ascending=False)
                        top_features = spread.head(8).index.tolist()
                        sep_table = pd.DataFrame({
                            "feature": spread.head(12).index,
                            "class_mean_std": spread.head(12).values
                        })

        corr_pairs = _top_correlations(work, num_cols[:40], top_k=12)

        # numeric summary (top N cols)
        summary_cols = num_cols[:20]
        numeric_summary = None
        if summary_cols:
            desc = work[summary_cols].describe().T
            desc = desc.rename(columns={
                "mean": "mean",
                "std": "std",
                "min": "min",
                "50%": "median",
                "max": "max",
            })
            keep = ["mean", "std", "min", "median", "max"]
            desc = desc[[c for c in keep if c in desc.columns]]
            desc = desc.reset_index().rename(columns={"index": "feature"})
            numeric_summary = desc

        return {
            "dashboard_type": "classification",
            "detected": {
                "target_col": schema.target_col,
                "numeric_cols_count": len(num_cols),
                "categorical_cols_count": len(schema.categorical_cols),
            },
            "kpis": [
                {"label": "Rows", "value": rows, "display": _format_int(rows)},
                {"label": "Columns", "value": cols, "display": _format_int(cols)},
                {"label": "Missing Cells", "value": missing_cells, "display": _format_int(missing_cells)},
                {"label": "Duplicate Rows", "value": dup_rows, "display": _format_int(dup_rows)},
                {"label": "Numeric Features", "value": len(num_cols), "display": _format_int(len(num_cols))},
                {"label": "Target", "value": target, "display": str(target) if target else "—"},
            ],
            "class_balance": class_balance,     # DataFrame: class,count
            "top_separation": sep_table,        # DataFrame
            "top_features": top_features,       # list[str]
            "corr_pairs": corr_pairs,           # DataFrame
            "numeric_summary": numeric_summary, # DataFrame
        }

    # ---- GENERIC MODE ----
    num_cols = [c for c in schema.numeric_cols if c in work.columns]
    corr_pairs = _top_correlations(work, num_cols[:40], top_k=12)

    cat_summaries = []
    for c in schema.categorical_cols[:3]:
        if c in work.columns:
            vc = _top_categorical_counts(work, c, n=10)
            cat_summaries.append({"col": c, "top_counts": vc})

    numeric_summary = None
    if num_cols:
        desc = work[num_cols[:20]].describe().T
        desc = desc.rename(columns={"50%": "median"}).reset_index().rename(columns={"index": "feature"})
        keep = ["feature", "mean", "std", "min", "median", "max"]
        numeric_summary = desc[[c for c in keep if c in desc.columns]]

    return {
        "dashboard_type": "generic",
        "detected": {
            "numeric_cols_count": len(num_cols),
            "categorical_cols_count": len(schema.categorical_cols),
            "date_col": schema.date_col,
        },
        "kpis": [
            {"label": "Rows", "value": rows, "display": _format_int(rows)},
            {"label": "Columns", "value": cols, "display": _format_int(cols)},
            {"label": "Missing Cells", "value": missing_cells, "display": _format_int(missing_cells)},
            {"label": "Duplicate Rows", "value": dup_rows, "display": _format_int(dup_rows)},
            {"label": "Numeric Columns", "value": len(num_cols), "display": _format_int(len(num_cols))},
            {"label": "Categorical Columns", "value": len(schema.categorical_cols), "display": _format_int(len(schema.categorical_cols))},
        ],
        "corr_pairs": corr_pairs,             # DataFrame
        "numeric_summary": numeric_summary,   # DataFrame
        "categorical_summaries": cat_summaries,  # list of {col, top_counts}
    }


# -----------------------------
# Executive Summary prompt builder (Sales only for now)
# -----------------------------
def build_exec_summary_prompt(spec: Dict) -> str:
    # Keep your existing summary prompt behavior for sales dashboards.
    # For non-sales dashboards, we'll return a simple generic prompt.
    dtype = spec.get("dashboard_type")

    if dtype != "sales":
        return (
            "You are an analyst. Write 3-5 concise bullet points summarizing the dashboard.\n"
            "If classification: mention class balance and top differentiating features.\n"
            "If generic: mention missingness, correlations, and key distributions.\n"
            f"Dashboard type: {dtype}\n"
        )

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
import re
import os
import json
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import dashboard_kpi as dk

# ----------------------------
# Setup
# ----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Data Copilot", layout="wide")
st.title("🤖 AI Data Analytics Copilot")
st.caption("Upload a File → ask in English → get analysis + chart + insights.")

# Session state: history + last run (so toggles don't wipe UI)
if "history" not in st.session_state:
    st.session_state.history = []

if "last_run" not in st.session_state:
    st.session_state.last_run = {
        "attempts": None,
        "has_result": False,
        "dataset_label": None,
        "question": None,
        "code": None,
        "result_df": None,
        "result_fig": None,
        "meta": None,
        "insights": None,
        "report_md": None,
    }

# ----------------------------
# Helpers
# ----------------------------
SYSTEM_PROMPT = """
You are a data analyst. Generate ONLY Python code using pandas/numpy/matplotlib.

Rules:
- A pandas DataFrame named df already exists.
- Do NOT read/write files.
- Do NOT use network calls.
- Do NOT import any modules.
- Put the final tabular answer in result_df (DataFrame) or None.
- Put the final chart in result_fig (matplotlib Figure) or None.
- If no chart is needed, set result_fig = None.
- Keep code under 30 lines.
- Do NOT print anything.
- Do NOT create new columns in df (e.g., df['x']=...) unless the user explicitly asks to create a new column.
- If the user asks for a column that doesn't exist, do NOT invent a replacement.
  Instead set:
    result_df = pd.DataFrame({"error":[<clear message>], "available_columns":[", ".join(df.columns)]})
    result_fig = None
"""

FIX_PROMPT = """
You are a data analyst and debugger. You previously generated Python code that failed.
Return ONLY corrected Python code.

Rules:
- A pandas DataFrame named df already exists.
- Do NOT read/write files.
- Do NOT use network calls.
- Do NOT import any modules.
- Put the final tabular answer in result_df (DataFrame) or None.
- Put the final chart in result_fig (matplotlib Figure) or None.
- If no chart is needed, set result_fig = None.
- Keep code under 30 lines.
- Do NOT print anything.
- Do NOT create new columns in df (e.g., df['x']=...) unless the user explicitly asks to create a new column.
- If the user asks for a column that doesn't exist, do NOT invent a replacement.
  Instead set:
    result_df = pd.DataFrame({"error":[<clear message>], "available_columns":[", ".join(df.columns)]})
    result_fig = None
"""

def agent_run(df: pd.DataFrame, question: str, max_retries: int = 2):
    attempts = []
    code = generate_pandas_code(df, question)

    for attempt in range(max_retries + 1):
        code_clean = re.sub(r"^import .*$", "", code, flags=re.MULTILINE)
        code_clean = re.sub(r"^from .*$", "", code_clean, flags=re.MULTILINE)
        code_clean = code_clean.strip()

        ok, reason = is_code_safe(code_clean)
        if not ok:
            attempts.append({
                "attempt": attempt + 1,
                "status": "blocked",
                "reason": reason,
                "code": code_clean
            })
            return None, None, code_clean, attempts, f"Blocked: {reason}"
        
        ok2, reason2 = validate_generated_code(code_clean, df)
        if not ok2:
            # Treat as an error so the fix-model rewrites it
            attempts.append({
                "attempt": attempt + 1,
                "status": "validation_error",
                "reason": reason2,
                "code": code_clean
            })

            if attempt < max_retries:
                code = fix_pandas_code(df, question, code_clean, f"ValidationError: {reason2}")
                continue

            return None, None, code_clean, attempts, f"Validation failed after retries: {reason2}"

        result_df, result_fig, err = run_generated_code(df, code_clean)

        if err is None:
            attempts.append({
                "attempt": attempt + 1,
                "status": "success",
                "code": code_clean
            })
            return result_df, result_fig, code_clean, attempts, None

        attempts.append({
            "attempt": attempt + 1,
            "status": "error",
            "error": err,
            "code": code_clean
        })

        if attempt < max_retries:
            code = fix_pandas_code(df, question, code_clean, err)

    return None, None, code_clean, attempts, "Execution failed after retries."

def llm_exec_summary(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Write concise executive bullet points. No fluff."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        parts = code.split("```")
        if len(parts) >= 3:
            code = parts[1]
        else:
            code = code.replace("```", "")
        code = code.strip()
        if code.startswith("python"):
            code = code[len("python"):].strip()
    return code.strip()


@st.cache_data(show_spinner=False)
def load_table(uploaded_file, sheet_name=None):
    # Streamlit file pointer can move; always reset.
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file type")

def detect_date_range(df: pd.DataFrame):
    """Try to find a date-like column and return (col_name, min_date, max_date) or (None, None, None)."""
    for c in df.columns:
        # Prefer obvious date columns
        name = str(c).lower()
        if any(k in name for k in ["date", "time", "timestamp", "order_date", "created", "dt"]):
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().sum() >= max(5, int(0.1 * len(df))):  # enough valid dates
                return c, s.min(), s.max()

    # Fallback: try any column that parses well
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().sum() >= max(5, int(0.2 * len(df))):
            return c, s.min(), s.max()

    return None, None, None


def phase2_summary_lite(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"ok": False, "error": "No data loaded."}

    rows, cols = df.shape

    # Duplicates
    dup_count = int(df.duplicated().sum())
    dup_pct = round((dup_count / rows) * 100, 2) if rows else 0.0

    # Missing %
    missing_pct = (df.isna().sum() / rows * 100).round(2) if rows else pd.Series(dtype=float)
    missing_sorted = missing_pct.sort_values(ascending=False)

    # Top missing columns (only > 0)
    top_missing = missing_sorted[missing_sorted > 0].head(5)
    top_missing_dict = {k: float(v) for k, v in top_missing.items()}

    # Date range (optional)
    date_col, dmin, dmax = detect_date_range(df)

    # --- Issues for scoring ---
    high_missing_20 = missing_sorted[missing_sorted >= 20]
    high_missing_30 = missing_sorted[missing_sorted >= 30]

    # Negative values in obvious money columns (profit/revenue/sales/etc.)
    money_like = [c for c in df.columns if any(k in str(c).lower() for k in ["profit", "revenue", "sales", "price", "amount"])]
    neg_issue = None
    for c in money_like[:8]:
        if pd.api.types.is_numeric_dtype(df[c]):
            neg = int((df[c] < 0).sum())
            if neg > 0:
                neg_issue = (c, neg)
                break

    # --- Health status (Green/Yellow/Red) ---
    # Red if: duplicates >=2% OR any column missing >=30% OR negative in profit/revenue present
    # Yellow if: duplicates >0 OR any column missing >=10%
    # Green otherwise
    any_missing_10 = bool((missing_sorted >= 10).any())
    is_red = (dup_pct >= 2.0) or (len(high_missing_30) > 0) or (neg_issue is not None)
    is_yellow = (dup_count > 0) or any_missing_10

    if is_red:
        status = "RED"
        status_text = "Risky"
    elif is_yellow:
        status = "YELLOW"
        status_text = "Needs Attention"
    else:
        status = "GREEN"
        status_text = "Healthy"

    # --- Warnings (max 3, grammar fixed) ---
    warnings = []

    if len(high_missing_20) > 0:
        n = len(high_missing_20)
        warnings.append(f"{n} column{'s' if n != 1 else ''} ha{'ve' if n != 1 else 's'} ≥ 20% missing values")

    if dup_count > 0:
        warnings.append(f"{dup_count} duplicate row{'s' if dup_count != 1 else ''} found ({dup_pct}%)")

    if neg_issue is not None:
        col, neg = neg_issue
        warnings.append(f"'{col}' has {neg} negative value{'s' if neg != 1 else ''}")

    if not warnings:
        warnings = ["Dataset looks healthy for analysis"]

    # --- Suggested questions (agent vibe) ---
    suggestions = []

    if date_col:
        # a safe, generic suggestion for time trend
        suggestions.append(f"Plot monthly trend using {date_col}")

    if top_missing_dict:
        worst_col = next(iter(top_missing_dict.keys()))
        suggestions.append(f"How many rows have missing {worst_col}?")

    if neg_issue is not None:
        col, _ = neg_issue
        suggestions.append(f"Show rows where {col} < 0")

    if len(suggestions) < 3:
        suggestions.append("Show summary statistics for key numeric columns")

    # Keep it short
    suggestions = suggestions[:3]

    return {
        "ok": True,
        "status": status,                 # GREEN / YELLOW / RED
        "status_text": status_text,       # Healthy / Needs Attention / Risky
        "overview": {
            "rows": int(rows),
            "columns": int(cols),
            "duplicate_rows": dup_count,
            "duplicate_percent": dup_pct,
            "date_col": date_col,
            "date_min": None if dmin is None or pd.isna(dmin) else str(dmin.date() if hasattr(dmin, "date") else dmin),
            "date_max": None if dmax is None or pd.isna(dmax) else str(dmax.date() if hasattr(dmax, "date") else dmax),
        },
        "warnings": warnings[:3],
        "top_missing_columns": top_missing_dict,
        "suggestions": suggestions,
    }



def dataset_profile(df: pd.DataFrame) -> dict:
    prof = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": [],
        "missing_top": df.isna().sum().sort_values(ascending=False).head(8).to_dict(),
        "numeric_cols": df.select_dtypes(include="number").columns.tolist()[:20],
        "object_cols": df.select_dtypes(include=["object"]).columns.tolist()[:20],
    }
    for c in df.columns[:30]:
        s = df[c]
        prof["columns"].append({
            "name": c,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "nunique": int(s.nunique(dropna=True)),
        })
    return prof

def validate_generated_code(code: str, df: pd.DataFrame) -> tuple[bool, str]:
    """
    Prevent the model from 'cheating' by creating new columns like df['dummy']=1.
    If it needs a derived column, user should explicitly ask for it.
    """
    existing = set(map(str, df.columns))

    if re.search(r"^\s*df\s*=", code, flags=re.MULTILINE):
        return False, "Model attempted to reassign df. Not allowed."

    # Detect df['col'] = ... assignments
    assigns = re.findall(r"df\[\s*['\"]([^'\"]+)['\"]\s*\]\s*=", code)
    for col in assigns:
        if col not in existing:
            return False, f"Model attempted to create new column '{col}'. Not allowed."

    # Detect df['col'] usage (reads)
    reads = re.findall(r"df\[\s*['\"]([^'\"]+)['\"]\s*\]", code)
    for col in reads:
        # allow new column check is already handled separately
        if col not in existing:
            return False, f"Model referenced missing column '{col}'."

    # Detect groupby('col') where col doesn't exist
    gbs = re.findall(r"\.groupby\(\s*['\"]([^'\"]+)['\"]\s*\)", code)
    for col in gbs:
        if col not in existing:
            return False, f"Model attempted to groupby missing column '{col}'."

    return True, ""

def is_code_safe(code: str) -> tuple[bool, str]:
    blocked_patterns = [
        r"\bimport\b", r"\bfrom\b",
        r"os\.", r"sys\.", r"subprocess", r"shutil",
        r"open\(", r"__import__", r"eval\(", r"exec\(",
        r"pickle", r"joblib",
        r"requests", r"http", r"socket",
        r"\bwrite\b", r"\bto_csv\b", r"\bto_excel\b",
    ]
    for pat in blocked_patterns:
        if re.search(pat, code):
            return False, f"Blocked unsafe pattern: `{pat}`"
    return True, ""


def generate_pandas_code(df: pd.DataFrame, question: str) -> str:
    prompt = f"""
Dataset columns: {list(df.columns)}
Sample rows (first 5): {df.head(5).to_dict(orient="records")}

User question: {question}

Return ONLY code.
"""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return strip_code_fences(resp.choices[0].message.content)

def fix_pandas_code(df: pd.DataFrame, question: str, bad_code: str, error_text: str) -> str:
    prompt = f"""
Dataset columns: {list(df.columns)}
Sample rows (first 5): {df.head(5).to_dict(orient="records")}

User question: {question}

The code below failed:
{bad_code}

Error traceback:
{error_text}

Return ONLY corrected code.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",  # 🔥 Stronger model for fixing
        messages=[
            {"role": "system", "content": FIX_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return strip_code_fences(resp.choices[0].message.content)

def generate_insights(question: str, result_df: pd.DataFrame) -> str:
    preview = result_df.head(12).to_dict(orient="records") if result_df is not None else []
    prompt = f"""
You are a business data analyst.

User question: {question}

Result preview (first rows):
{preview}

Write:
- 3 concise insights (bullet points)
- 1 recommendation
- 1 caveat/assumption

Be clear and non-technical.
"""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def analyze_query_and_risks(question: str, df: pd.DataFrame, code: str,
                            result_df: pd.DataFrame | None) -> dict:
    prof = dataset_profile(df)
    result_preview = []
    if isinstance(result_df, pd.DataFrame):
        result_preview = result_df.head(10).to_dict(orient="records")

    prompt = f"""
Return ONLY valid JSON.

User question:
{question}

Dataset profile:
{json.dumps(prof)}

Generated code:
{code}

Result preview:
{json.dumps(result_preview)}

Tasks:
1) Query complexity:
   - label: "Basic" | "Intermediate" | "Advanced"
   - operations: list like ["aggregation","groupby","time-series","correlation","plotting","ranking","filtering"]
   - confidence_score: 0-100

2) Bias & risk detector (dataset + analysis):
   - dataset_risks: list
   - analysis_risks: list
   - mitigations: list

JSON schema:
{{
  "complexity": {{
    "label": "...",
    "operations": ["..."],
    "confidence_score": 0
  }},
  "bias_risk": {{
    "dataset_risks": ["..."],
    "analysis_risks": ["..."],
    "mitigations": ["..."]
  }}
}}
"""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except Exception:
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)


def run_generated_code(df: pd.DataFrame, code: str):
    safe_builtins = {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "range": range,
        "sorted": sorted,
        "round": round,
        "enumerate": enumerate,
        "zip": zip,
    }

    local_env = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "result_df": None,
        "result_fig": None,
    }

    try:
        exec(code, {"__builtins__": safe_builtins}, local_env)
        return local_env.get("result_df"), local_env.get("result_fig"), None
    except Exception:
        return None, None, traceback.format_exc()


def build_markdown_report(dataset_name: str, question: str, code: str,
                          result_df: pd.DataFrame | None,
                          insights: str | None,
                          meta: dict | None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = []
    md.append("# AI Data Analytics Copilot Report\n")
    md.append(f"**Generated:** {ts}\n")
    md.append(f"**Dataset:** {dataset_name}\n")
    md.append(f"**Question:** {question}\n")

    if meta:
        md.append("## Query Complexity\n")
        md.append(f"- Level: {meta['complexity']['label']}\n")
        md.append(f"- Confidence: {meta['complexity']['confidence_score']}%\n")
        md.append(f"- Operations: {', '.join(meta['complexity']['operations'])}\n\n")

        md.append("## Bias & Risk Detector\n")
        md.append("**Dataset risks:**\n")
        for r in meta["bias_risk"]["dataset_risks"]:
            md.append(f"- {r}\n")
        md.append("\n**Analysis risks:**\n")
        for r in meta["bias_risk"]["analysis_risks"]:
            md.append(f"- {r}\n")
        md.append("\n**Mitigations:**\n")
        for r in meta["bias_risk"]["mitigations"]:
            md.append(f"- {r}\n")
        md.append("\n")

    md.append("## Generated Code\n")
    md.append("```python\n" + (code or "").strip() + "\n```\n")

    md.append("## Result (Preview)\n")
    if result_df is not None and isinstance(result_df, pd.DataFrame):
        md.append(result_df.head(20).to_markdown(index=False))
        md.append("\n")
    else:
        md.append("_No table result produced._\n")

    md.append("## AI Insights\n")
    md.append(insights if insights else "_No insights generated._")
    md.append("\n")

    md.append("## Notes\n")
    md.append("- Results depend on data quality and column meanings.\n")
    md.append("- AI-generated code may require validation.\n")

    return "\n".join(md)


def render_executive_dashboard(df: pd.DataFrame, llm_callable=None):
    st.subheader("📊 Executive Dashboard")

    schema = dk.detect_schema(df)
    spec = dk.compute_dashboard(df, schema)

    # Keep UI clean: detected info is tucked away
    with st.expander("What the agent detected (columns used)", expanded=False):
        st.json(spec.get("detected", {}))

    # KPI Cards (executive-style: 4 cards)
    kpis = spec.get("kpis", [])
    top_kpis = [k for k in kpis if k["label"] in ("Total Revenue", "Total Profit", "Profit Margin", "Avg Discount")]
    cols = st.columns(4)
    for i, k in enumerate(top_kpis[:4]):
        cols[i].metric(k["label"], k["display"])

    # --- Transparency: show what columns the agent used (clean + exec-friendly)
    det = spec.get("detected", {})
    cat0 = (det.get("category_cols") or ["—"])[0]

    st.caption(
        f"Using: date=`{det.get('used_date_col')}` | "
        f"revenue=`{det.get('revenue_col')}` | "
        f"profit=`{det.get('profit_col')}` | "
        f"category=`{cat0}`"
    )

    det = spec.get("detected", {})
    

    st.divider()

    ins = spec.get("exec_insights", {})
    if ins:        
        rev_mom = ins.get("revenue_mom_pct")
        prof_mom = ins.get("profit_mom_pct")
        rev_all = ins.get("revenue_overall_pct")
        prof_all = ins.get("profit_overall_pct")

        def pct(x):
            return "—" if x is None else f"{x*100:.1f}%"

        st.caption(
            f"MoM Revenue: {pct(rev_mom)} | MoM Profit: {pct(prof_mom)}  •  "
            f"Overall Revenue: {pct(rev_all)} | Overall Profit: {pct(prof_all)}"
        )

    st.divider()

    # Charts (only show if available)
    left, right = st.columns([1.2, 1])

    trend_df = spec.get("trend")
    if trend_df is not None and not trend_df.empty:
        with left:
            st.markdown("### Revenue Trend")
            st.line_chart(trend_df.set_index("period")["revenue"])

    profit_trend_df = spec.get("profit_trend")
    if profit_trend_df is not None and not profit_trend_df.empty:
        with left:
            st.markdown("### Profit Trend")
            st.line_chart(profit_trend_df.set_index("period")["profit"])

    breakdown_df = spec.get("breakdown")
    if breakdown_df is not None and not breakdown_df.empty:
        with right:
            st.markdown("### Top Categories by Revenue")
            st.bar_chart(breakdown_df.set_index("category")["revenue"])

    profit_breakdown_df = spec.get("profit_breakdown")
    if profit_breakdown_df is not None and not profit_breakdown_df.empty:
        with right:
            st.markdown("### Top Categories by Profit")
            st.bar_chart(profit_breakdown_df.set_index("category")["profit"])

    profit_df = spec.get("profit_breakdown")
    if profit_df is not None and not profit_df.empty:
        st.markdown("### Top Categories by Profit")
        st.bar_chart(profit_df.set_index("category")["profit"])

    st.divider()

    # Executive Summary
    st.markdown("### Executive Summary")
    prompt = dk.build_exec_summary_prompt(spec)

    summary_text = None
    if llm_callable is not None:
        try:
            summary_text = llm_callable(prompt)
        except Exception:
            summary_text = None

    if not summary_text:
        summary_text = (
            "- KPIs are shown above.\n"
            "- Trend and category charts appear when a valid date/category column is detected.\n"
            "- (Optional) Wire LLM summary to generate a true executive narrative."
        )

    st.markdown(summary_text)

    # --- Download dashboard spec (agent output)
    with st.expander("⚙️ Dashboard Output (Advanced)"):
        st.download_button(
            label="⬇️ Download Dashboard Spec (JSON)",
            data=json.dumps(spec, default=str, indent=2).encode("utf-8"),
            file_name="executive_dashboard_spec.json",
            mime="application/json",
            key="download_exec_dashboard_spec_json",
        )

    return spec

def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df.columns are unique by appending .1, .2, ... to duplicates.
    Also trims whitespace to avoid 'col' vs 'col ' issues.
    """
    df = df.copy()
    cols = [str(c).strip() for c in df.columns]  # trim spaces
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

# ----------------------------
# UI
# ----------------------------


with st.sidebar:
    st.header("⚙️ Controls")

    # ----------------------------
    # Persist defaults (set once)
    # ----------------------------
    if "mode" not in st.session_state:
        st.session_state.mode = "Overview"

    # View on/off toggles (NOT inside Advanced)
    if "enable_overview" not in st.session_state:
        st.session_state.enable_overview = True
    if "enable_preview" not in st.session_state:
        st.session_state.enable_preview = True
    if "enable_quality" not in st.session_state:
        st.session_state.enable_quality = True
    if "enable_kpi" not in st.session_state:
        st.session_state.enable_kpi = True
    if "enable_askai" not in st.session_state:
        st.session_state.enable_askai = True
    if "enable_history" not in st.session_state:
        st.session_state.enable_history = True

    # AI output settings (only insights default ON)
    if "show_code" not in st.session_state:
        st.session_state.show_code = False
    if "show_meta" not in st.session_state:
        st.session_state.show_meta = False
    if "generate_insight_toggle" not in st.session_state:
        st.session_state.generate_insight_toggle = True  # ✅ default ON

    # ----------------------------
    # View dropdown (ALWAYS shows all)
    # ----------------------------
    all_views = ["Overview", "Data Preview", "Data Quality", "KPI Dashboard", "Ask AI", "History"]

    st.session_state.mode = st.selectbox(
        "View",
        all_views,
        index=all_views.index(st.session_state.mode) if st.session_state.mode in all_views else 0
    )
    mode = st.session_state.mode  # keep rest of app unchanged

    st.divider()

    # ----------------------------
    # On/Off toggles (right below View)
    # ----------------------------
    st.markdown("**Enable views**")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.enable_overview = st.toggle("Overview", value=st.session_state.enable_overview)
        st.session_state.enable_quality = st.toggle("Data Quality", value=st.session_state.enable_quality)
        st.session_state.enable_askai = st.toggle("Ask AI", value=st.session_state.enable_askai)
    with c2:
        st.session_state.enable_preview = st.toggle("Data Preview", value=st.session_state.enable_preview)
        st.session_state.enable_kpi = st.toggle("KPI Dashboard", value=st.session_state.enable_kpi)
        st.session_state.enable_history = st.toggle("History", value=st.session_state.enable_history)

    # If user is currently on a disabled view, bounce to Overview
    enabled_map = {
        "Overview": st.session_state.enable_overview,
        "Data Preview": st.session_state.enable_preview,
        "Data Quality": st.session_state.enable_quality,
        "KPI Dashboard": st.session_state.enable_kpi,
        "Ask AI": st.session_state.enable_askai,
        "History": st.session_state.enable_history,
    }
    if not enabled_map.get(st.session_state.mode, True):
        st.warning(f"'{st.session_state.mode}' is turned OFF. Switching to Overview.")
        st.session_state.mode = "Overview"
        mode = "Overview"

    st.divider()

    # ----------------------------
    # Advanced: AI output settings only
    # ----------------------------
    with st.expander("Advanced (AI output settings)", expanded=False):
        st.session_state.generate_insight_toggle = st.toggle(
            "Generate AI insights",
            value=st.session_state.generate_insight_toggle,  # ✅ default ON
        )
        # If you truly want ONLY insights, comment these out:
        st.session_state.show_code = st.toggle("Show generated code", value=st.session_state.show_code)
        st.session_state.show_meta = st.toggle("Show complexity + bias/risk", value=st.session_state.show_meta)

    st.divider()

    # ----------------------------
    # History quick actions (optional)
    # ----------------------------
    st.subheader("🧾 History")
    st.write(f"Saved runs: **{len(st.session_state.history)}**")
    if st.button("Clear history"):
        st.session_state.history = []
        st.success("History cleared.")

# Make these available to the rest of your app (same variable names as before)
show_code = st.session_state.show_code
show_meta = st.session_state.show_meta
generate_insight_toggle = st.session_state.generate_insight_toggle

uploaded = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls"])

sheet = None
dataset_label = None
df = None

if uploaded:
    dataset_label = uploaded.name

    # Excel: sheet dropdown
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        uploaded.seek(0)
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox("Select sheet", xls.sheet_names)
        dataset_label = f"{uploaded.name} / {sheet}"

    df = load_table(uploaded, sheet_name=sheet)

    if mode == "Overview":
        st.subheader("✅ Dataset Loaded")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        st.info("Use the sidebar to open Data Preview, Data Quality, KPI Dashboard, or Ask AI.")

    # Detect duplicates BEFORE fixing
    raw_cols = [str(c).strip() for c in df.columns]
    dupes = pd.Index(raw_cols)[pd.Index(raw_cols).duplicated()].tolist()

    # Fix columns
    df = make_unique_columns(df)
    if not df.columns.is_unique:
        st.error("Columns are STILL not unique after cleaning (unexpected).")
        st.write("Duplicates:", df.columns[df.columns.duplicated()].tolist())
        st.write(df.columns.tolist())

    # Warn user if needed
    if dupes:
        st.warning(f"Duplicate column names detected and auto-renamed: {sorted(set(dupes))}")
    
    if mode == "Data Preview":
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

    # ----------------------------
    # Phase 2 (Lite): Data Quality Summary (3 blocks)
    # ----------------------------
    if mode == "Data Quality":
        if mode == "Data Quality":
            st.subheader("🧼 Data Quality Summary")
            p2 = phase2_summary_lite(df)

        if not p2.get("ok"):
            st.info(p2.get("error", "No summary available."))
        else:
            ov = p2["overview"]

            # Status badge (Green/Yellow/Red) + legend
            status = p2["status"]
            status_text = p2["status_text"]

            if status == "GREEN":
                st.success(f"Status: {status} — {status_text}")
            elif status == "YELLOW":
                st.warning(f"Status: {status} — {status_text}")
            else:
                st.error(f"Status: {status} — {status_text}")

            st.caption("Legend: 🟢 GREEN = Healthy   |   🟡 YELLOW = Needs Attention   |   🔴 RED = Risky")

            # BLOCK 1: Dataset Overview
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{ov['rows']:,}")
            c2.metric("Columns", f"{ov['columns']:,}")
            c3.metric("Duplicates", f"{ov['duplicate_rows']:,}")
            c4.metric("Duplicate %", f"{ov['duplicate_percent']}%")

            if ov.get("date_col") and ov.get("date_min") and ov.get("date_max"):
                st.caption(f"Date range detected from **{ov['date_col']}**: {ov['date_min']} → {ov['date_max']}")

            # BLOCK 2: Top Warnings
            st.markdown("### ⚠️ Data Health Warnings")
            for w in p2["warnings"]:
                if "healthy" in w.lower():
                    st.success(w)
                else:
                    # Keep warnings consistent with status severity
                    if status == "RED":
                        st.error(w)
                    elif status == "YELLOW":
                        st.warning(w)
                    else:
                        st.info(w)

            # BLOCK 3: Missing Values (Top 5)
            st.markdown("### 🕳️ Missing Values (Top 5)")
            miss = p2["top_missing_columns"]
            if not miss:
                st.success("No missing values detected.")
            else:
                miss_df = pd.DataFrame(
                    [{"Column": k, "Missing %": v} for k, v in miss.items()]
                ).sort_values("Missing %", ascending=False)
                st.dataframe(miss_df, use_container_width=True, hide_index=True)

            # Suggested questions (optional but impressive)
            st.markdown("### 💡 Suggested Questions")
            for s in p2.get("suggestions", []):
                st.write(f"- {s}")

    # ----------------------------
    # Phase 3: KPI Auto Executive Dashboard
    # ----------------------------
    if mode == "KPI Dashboard":
        spec = render_executive_dashboard(df, llm_callable=llm_exec_summary)
        st.session_state.last_run = {**st.session_state.last_run, "kpi_spec": spec}
        

    st.divider()
    if mode == "Ask AI":
        st.subheader("💬 Ask a question about your data")
        question = st.text_input("Example: Plot monthly revenue trend")

        colA, colB = st.columns([1, 1])
        with colA:
            run_btn = st.button("Run AI Analysis", type="primary")
        with colB:
            st.caption("Tip: ask for a table or a chart (or both).")

        # -------- Run analysis (compute + STORE only) --------
        if run_btn:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    with st.spinner("Running agentic analysis (self-healing)..."):
                        result_df, result_fig, code, attempts, agent_err = agent_run(df, question, max_retries=2)

                    if agent_err:
                        st.error(agent_err)
                        with st.expander("🛠 Agent Attempt Logs"):
                            st.json(attempts)
                    else:
                        meta = None
                        if show_meta:
                            with st.spinner("Assessing complexity + bias/risk..."):
                                meta = analyze_query_and_risks(question, df, code, result_df)

                        insights_text = None
                        if generate_insight_toggle and isinstance(result_df, pd.DataFrame):
                            with st.spinner("Generating insights..."):
                                insights_text = generate_insights(question, result_df)

                        report_md = build_markdown_report(
                            dataset_name=dataset_label,
                            question=question,
                            code=code,
                            result_df=result_df if isinstance(result_df, pd.DataFrame) else None,
                            insights=insights_text,
                            meta=meta
                        )

                        #  Persist results for reruns (toggles won't clear output)
                        st.session_state.last_run = {
                            "has_result": True,
                            "dataset_label": dataset_label,
                            "question": question,
                            "code": code,
                            "result_df": result_df,
                            "result_fig": result_fig,
                            "meta": meta,
                            "insights": insights_text,
                            "report_md": report_md,
                            "attempts": attempts,   
                        }

                        # Save to history
                        st.session_state.history.append({
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "question": question,
                            "code": code,
                            "insights": insights_text,
                            "result_preview": result_df.head(20).copy() if isinstance(result_df, pd.DataFrame) else None,
                            "dataset": dataset_label,
                        })

                except Exception as e:
                    st.error(f"Error: {e}")

# -------- Render persisted outputs (always) --------
lr = st.session_state.last_run
if lr.get("has_result"):
    st.divider()
    st.subheader("✅ Latest Result")
    st.caption(f"Dataset: {lr['dataset_label']}")

    if lr.get("attempts"):
        with st.expander("🛠 Agent Attempts"):
            st.json(lr["attempts"])

    if show_code and lr.get("code"):
        st.subheader("🧠 Generated Code")
        st.code(lr["code"], language="python")

    if show_meta and lr.get("meta"):
        meta = lr["meta"]
        st.subheader("🧭 Query Complexity")
        st.write(
            f"**Level:** {meta['complexity']['label']}  |  "
            f"**Confidence:** {meta['complexity']['confidence_score']}%"
        )
        st.write("**Operations:**", ", ".join(meta["complexity"]["operations"]))

        st.subheader("⚠️ Bias & Risk Detector")
        st.markdown("**Dataset risks**")
        st.write(meta["bias_risk"]["dataset_risks"])
        st.markdown("**Analysis risks**")
        st.write(meta["bias_risk"]["analysis_risks"])
        st.markdown("**Mitigations**")
        st.write(meta["bias_risk"]["mitigations"])

    if lr.get("result_df") is not None:
        st.subheader("📋 Result Table")
        st.dataframe(lr["result_df"], use_container_width=True)

    if lr.get("result_fig") is not None:
        st.subheader("📈 Chart")
        st.pyplot(lr["result_fig"], clear_figure=True)

    if generate_insight_toggle and lr.get("insights"):
        st.subheader("🧠 AI Insights")
        st.write(lr["insights"])

    if lr.get("report_md"):
        st.download_button(
            label="⬇️ Download report (Markdown)",
            data=lr["report_md"].encode("utf-8"),
            file_name="ai_data_copilot_report.md",
            mime="text/markdown",
        )

# -------- History viewer --------
if st.session_state.history:
    st.divider()
    if mode == "History":
        st.subheader("🧾 Run History (latest first)")
        for item in reversed(st.session_state.history[-10:]):
            with st.expander(f"{item['time']} — {item['question']}"):
                st.write(f"**Dataset:** {item['dataset']}")
                if show_code:
                    st.code(item["code"], language="python")
                if item["result_preview"] is not None:
                    st.dataframe(item["result_preview"], use_container_width=True)
                if item["insights"]:
                    st.markdown("**Insights:**")
                    st.write(item["insights"])

import re
import os
import json
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import dashboard_kpi as dk

# ----------------------------
# Setup
# ----------------------------
load_dotenv()

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=api_key)

if not api_key:
    st.error("OpenAI API key is missing. Please configure OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

st.set_page_config(page_title="AI Data Copilot", layout="wide")

st.title("🤖 AI Data Analyst Copilot")

st.markdown("""

This app can:
- 🔎 Explore dataset structure and quality
- 📊 Automatically generate KPI dashboards
- 💬 Answer questions about your data using AI
- 📈 Provide executive insights

⚠️ Please do not upload sensitive or confidential data.
""")

# ---- Custom CSS: clean dark-accent theme ----
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2d3148;
        border-radius: 10px;
        padding: 14px 18px;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1d27;
        border-radius: 8px;
        padding: 6px 18px;
        color: #8888aa;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #3d5afe !important;
        color: white !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #13161f; }
    /* Dividers */
    hr { border-color: #2d3148; }
    /* Section headers */
    h3 { color: #c8d0e7; }
    /* Dataframes */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

st.caption("Upload a file → explore → ask anything → get instant analysis.")

# ----------------------------
# Session state
# ----------------------------
for key, default in [
    ("history", []),
    ("last_run", {
        "has_result": False, "dataset_label": None, "question": None,
        "code": None, "result_df": None, "result_fig": None,
        "meta": None, "insights": None, "report_md": None, "attempts": None,
        "text_answer": None,
    }),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------
# Prompts
# ----------------------------
SYSTEM_PROMPT = """
You are an expert data analyst. Generate Python code using pandas, numpy, matplotlib,
scipy, sklearn, or any other standard library as needed.

A pandas DataFrame named `df` already exists in scope.
Also available: pd, np, plt, scipy, sklearn (if installed).

Rules:
- Do NOT read or write files.
- Do NOT make network calls.
- Do NOT use os, sys, subprocess, or shell commands.
- Put the final tabular answer in `result_df` (a DataFrame) or set it to None.
- Put the final chart in `result_fig` (a matplotlib Figure) or set it to None.
- Do NOT print anything; use result_df / result_fig for output.
- If a column the user asks about does not exist, set:
    result_df = pd.DataFrame({"error": ["Column not found"], "available_columns": [", ".join(df.columns)]})
    result_fig = None
"""

FIX_PROMPT = """
You are an expert data analyst and debugger.
The Python code below failed. Return ONLY corrected code.

A pandas DataFrame named `df` already exists.
Available: pd, np, plt (matplotlib.pyplot).

Rules: no file I/O, no network, no os/sys/subprocess.
Output goes to result_df (DataFrame or None) and result_fig (matplotlib Figure or None).
"""

# ----------------------------
# Helpers
# ----------------------------
def strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        parts = code.split("```")
        code = parts[1] if len(parts) >= 3 else code.replace("```", "")
        code = code.strip()
        if code.startswith("python"):
            code = code[len("python"):].strip()
    return code.strip()


@st.cache_data(show_spinner=False)
def load_table(uploaded_file, sheet_name=None):
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, sheet_name=sheet_name)
    raise ValueError("Unsupported file type")


def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [str(c).strip() for c in df.columns]
    seen: dict = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0; new_cols.append(c)
        else:
            seen[c] += 1; new_cols.append(f"{c}.{seen[c]}")
    df.columns = new_cols
    return df


def dataset_profile(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": [{"name": c, "dtype": str(df[c].dtype),
                     "missing": int(df[c].isna().sum()),
                     "nunique": int(df[c].nunique(dropna=True))}
                    for c in df.columns[:30]],
        "missing_top": df.isna().sum().sort_values(ascending=False).head(8).to_dict(),
        "numeric_cols": df.select_dtypes(include="number").columns.tolist()[:20],
        "object_cols":  df.select_dtypes(include="object").columns.tolist()[:20],
    }


def is_code_safe(code: str):
    blocked = [r"os\.", r"sys\.", r"subprocess", r"shutil",
               r"open\(", r"__import__", r"eval\(", r"exec\(",
               r"pickle", r"requests", r"http", r"socket",
               r"\bwrite\b", r"\bto_csv\b", r"\bto_excel\b"]
    for pat in blocked:
        if re.search(pat, code):
            return False, f"Blocked unsafe pattern: `{pat}`"
    return True, ""


def run_generated_code(df: pd.DataFrame, code: str):
    import matplotlib.pyplot as _plt
    safe_builtins = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k, None)
                     for k in ["len","min","max","sum","abs","range","sorted","round","enumerate","zip","list","dict","set","tuple","str","int","float","bool","print"]}
    # provide scipy + sklearn if available
    extra = {}
    try:
        import scipy; extra["scipy"] = scipy
    except ImportError: pass
    try:
        import sklearn; extra["sklearn"] = sklearn
    except ImportError: pass

    local_env = {"df": df.copy(), "pd": pd, "np": np, "plt": _plt,
                 "result_df": None, "result_fig": None, **extra}
    try:
        exec(code, {"__builtins__": safe_builtins}, local_env)
        return local_env.get("result_df"), local_env.get("result_fig"), None
    except Exception:
        return None, None, traceback.format_exc()


def generate_pandas_code(df: pd.DataFrame, question: str) -> str:
    prompt = f"""Dataset columns: {list(df.columns)}
Sample rows (first 5): {df.head(5).to_dict(orient='records')}

User question: {question}

Return ONLY Python code."""
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": prompt}],
        temperature=0.2,
    )
    return strip_code_fences(resp.choices[0].message.content)


def fix_pandas_code(df: pd.DataFrame, question: str, bad_code: str, error_text: str) -> str:
    prompt = f"""Dataset columns: {list(df.columns)}
Sample rows (first 5): {df.head(5).to_dict(orient='records')}

User question: {question}

Failed code:
{bad_code}

Error:
{error_text}

Return ONLY corrected code."""
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": FIX_PROMPT},
                  {"role": "user",   "content": prompt}],
        temperature=0.2,
    )
    return strip_code_fences(resp.choices[0].message.content)


def is_analytical_question(question: str, df: pd.DataFrame) -> bool:
    """Ask GPT whether this question needs code/data analysis or is a plain text question."""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Answer only YES or NO."},
            {"role": "user", "content":
             f"The user has uploaded a dataset with columns: {list(df.columns)}.\n"
             f"Does this question require data analysis, computation, aggregation, or chart generation?\n"
             f"Question: {question}\nAnswer YES or NO."}
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip().upper().startswith("Y")


def answer_text_question(question: str, df: pd.DataFrame) -> str:
    """Answer a non-analytical question using dataset context."""
    sample = df.head(5).to_dict(orient="records")
    col_info = [{"col": c, "dtype": str(df[c].dtype), "sample": str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else ""}
                for c in df.columns[:20]]
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content":
             "You are a helpful data analyst assistant. Answer questions about the dataset clearly and concisely. "
             "Use the column names, types, and sample data provided to give a grounded, accurate answer."},
            {"role": "user", "content":
             f"Dataset overview:\nColumns: {json.dumps(col_info)}\nSample rows: {json.dumps(sample)}\n\nQuestion: {question}"}
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def agent_run(df: pd.DataFrame, question: str, max_retries: int = 3):
    attempts = []
    code = generate_pandas_code(df, question)

    for attempt in range(max_retries + 1):
        # Strip import lines (model may add them; we re-inject via exec env)
        code_clean = re.sub(r"^import (?!pandas|numpy|matplotlib|scipy|sklearn).*$", "", code, flags=re.MULTILINE)
        code_clean = code_clean.strip()

        ok, reason = is_code_safe(code_clean)
        if not ok:
            attempts.append({"attempt": attempt+1, "status": "blocked", "reason": reason, "code": code_clean})
            return None, None, code_clean, attempts, f"Blocked: {reason}"

        result_df, result_fig, err = run_generated_code(df, code_clean)
        if err is None:
            attempts.append({"attempt": attempt+1, "status": "success", "code": code_clean})
            return result_df, result_fig, code_clean, attempts, None

        attempts.append({"attempt": attempt+1, "status": "error", "error": err, "code": code_clean})
        if attempt < max_retries:
            code = fix_pandas_code(df, question, code_clean, err)

    return None, None, code_clean, attempts, "Execution failed after retries."


def generate_insights(question: str, result_df: pd.DataFrame) -> str:
    preview = result_df.head(12).to_dict(orient="records") if result_df is not None else []
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content":
                   f"You are a business data analyst.\nQuestion: {question}\nResult: {preview}\n"
                   "Write: 3 concise insights (bullets), 1 recommendation, 1 caveat. Be clear, non-technical."}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def llm_exec_summary(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": "Write concise executive bullet points. No fluff."},
                  {"role": "user",   "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def generate_ai_quality_report(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    missing = df.isna().sum()
    missing_pct = (missing / rows * 100).round(2)
    top_missing = missing_pct[missing_pct > 0].sort_values(ascending=False).head(8).to_dict()
    dup_count = int(df.duplicated().sum())
    dtypes = df.dtypes.astype(str).to_dict()
    numeric_stats = df.describe().to_dict() if not df.select_dtypes(include="number").empty else {}

    profile_txt = (
        f"Rows: {rows}, Columns: {cols}\n"
        f"Duplicate rows: {dup_count} ({round(dup_count/rows*100,2)}%)\n"
        f"Columns with missing values: {json.dumps(top_missing)}\n"
        f"Column dtypes: {json.dumps(dtypes)}\n"
        f"Numeric stats (describe): {json.dumps(numeric_stats, default=str)}"
    )
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content":
             "You are a senior data quality analyst. Given a dataset profile, write a clear quality report. "
             "Structure it as: (1) a metrics summary section, then (2) a narrative analysis. "
             "The narrative should interpret what the numbers mean, flag risks, and give 2-3 actionable recommendations. "
             "Use markdown formatting (headers, bullets). Be direct and specific."},
            {"role": "user", "content": f"Dataset profile:\n{profile_txt}"}
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def build_markdown_report(dataset_name, question, code, result_df, insights, meta=None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = [f"# AI Data Analyst Copilot Report\n",
          f"**Generated:** {ts}\n**Dataset:** {dataset_name}\n**Question:** {question}\n"]
    md.append("## Generated Code\n```python\n" + (code or "").strip() + "\n```\n")
    md.append("## Result\n")
    if isinstance(result_df, pd.DataFrame):
        md.append(result_df.head(20).to_markdown(index=False) + "\n")
    else:
        md.append("_No table result._\n")
    md.append(f"## AI Insights\n{insights or '_None._'}\n")
    return "\n".join(md)


# ----------------------------
# Plotly chart helpers
# ----------------------------
PLOTLY_TEMPLATE = "plotly_dark"

def plotly_line(df: pd.DataFrame, x: str, y: str, title: str, color: str = None):
    fig = px.line(df, x=x, y=y, color=color, title=title, template=PLOTLY_TEMPLATE,
                  markers=True)
    fig.update_layout(hovermode="x unified", legend_title_text="")
    return fig

def plotly_bar(df: pd.DataFrame, x: str, y: str, title: str, orientation="v", color: str = None):
    fig = px.bar(df, x=x, y=y, color=color, title=title, template=PLOTLY_TEMPLATE,
                 orientation=orientation, text_auto=True)
    fig.update_traces(textfont_size=11)
    return fig

def plotly_pie(df: pd.DataFrame, names: str, values: str, title: str):
    fig = px.pie(df, names=names, values=values, title=title, template=PLOTLY_TEMPLATE,
                 hole=0.35)
    fig.update_traces(textinfo="percent+label")
    return fig

def plotly_corr_heatmap(corr_pairs: pd.DataFrame):
    # pivot back to matrix
    all_feats = sorted(set(corr_pairs["feature_a"].tolist() + corr_pairs["feature_b"].tolist()))
    mat = pd.DataFrame(np.eye(len(all_feats)), index=all_feats, columns=all_feats)
    for _, row in corr_pairs.iterrows():
        mat.loc[row["feature_a"], row["feature_b"]] = row["corr"]
        mat.loc[row["feature_b"], row["feature_a"]] = row["corr"]
    fig = px.imshow(mat, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Correlation Heatmap", template=PLOTLY_TEMPLATE,
                    aspect="auto")
    return fig


# ----------------------------
# KPI Dashboard renderer
# ----------------------------
def render_kpi_dashboard(df: pd.DataFrame, llm_callable=None):
    schema = dk.detect_schema(df)
    spec   = dk.compute_dashboard(df, schema)
    dtype  = spec.get("dashboard_type", "generic")
    raw    = spec.get("raw", df)

    # ---- Minimal column check ----
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(df.columns) < 2:
        st.warning("⚠️ Dataset has too few columns to generate a meaningful dashboard.")
        return

    # ---- KPI Cards ----
    kpis = spec.get("kpis", [])
    display_kpis = [k for k in kpis if k["display"] != "—"][:6]
    if display_kpis:
        cols_ui = st.columns(len(display_kpis))
        for i, k in enumerate(display_kpis):
            cols_ui[i].metric(k["label"], k["display"])
    st.divider()

    # ==========================
    # SALES DASHBOARD
    # ==========================
    if dtype == "sales":
        det = spec.get("detected", {})
        category_cols = det.get("category_cols") or []

        # --- Filters ---
        with st.expander("🔧 Filters", expanded=False):
            filter_col = st.selectbox("Filter by category column", ["(none)"] + category_cols, key="kpi_filter_col")
            filter_val = None
            if filter_col != "(none)":
                unique_vals = sorted(raw[filter_col].dropna().astype(str).unique().tolist())
                filter_val  = st.multiselect(f"Select {filter_col} values", unique_vals, default=unique_vals[:], key="kpi_filter_val")

        # Apply filter
        filtered = raw.copy()
        if filter_col != "(none)" and filter_val:
            filtered = filtered[filtered[filter_col].astype(str).isin(filter_val)]

        # Recompute spec on filtered data if needed
        if filter_col != "(none)" and filter_val:
            fschema = dk.detect_schema(filtered)
            fspec   = dk.compute_dashboard(filtered, fschema)
            trend_df          = fspec.get("trend")
            profit_trend_df   = fspec.get("profit_trend")
            breakdown_df      = fspec.get("breakdown")
            profit_bdown_df   = fspec.get("profit_breakdown")
            drill_df          = fspec.get("drill_data")
        else:
            trend_df          = spec.get("trend")
            profit_trend_df   = spec.get("profit_trend")
            breakdown_df      = spec.get("breakdown")
            profit_bdown_df   = spec.get("profit_breakdown")
            drill_df          = spec.get("drill_data")

        # --- Charts ---
        col_l, col_r = st.columns(2)

        if isinstance(trend_df, pd.DataFrame) and not trend_df.empty:
            with col_l:
                fig = plotly_line(trend_df, "period", "revenue", "📈 Revenue Trend")
                st.plotly_chart(fig, use_container_width=True)

        if isinstance(profit_trend_df, pd.DataFrame) and not profit_trend_df.empty:
            with col_r:
                fig = plotly_line(profit_trend_df, "period", "profit", "📈 Profit Trend")
                st.plotly_chart(fig, use_container_width=True)

        if isinstance(breakdown_df, pd.DataFrame) and not breakdown_df.empty:
            with col_l:
                fig = plotly_bar(breakdown_df, "revenue", "category",
                                 "🏆 Top Categories by Revenue", orientation="h")
                st.plotly_chart(fig, use_container_width=True)

        if isinstance(profit_bdown_df, pd.DataFrame) and not profit_bdown_df.empty:
            with col_r:
                fig = plotly_bar(profit_bdown_df, "profit", "category",
                                 "💰 Top Categories by Profit", orientation="h")
                st.plotly_chart(fig, use_container_width=True)

        # --- Drill-down ---
        if isinstance(drill_df, pd.DataFrame) and not drill_df.empty:
            st.markdown("### 🔍 Revenue Drill-down by Category")
            avail_cats = sorted(drill_df["category"].astype(str).unique().tolist())
            selected_cats = st.multiselect("Select categories to compare", avail_cats,
                                           default=avail_cats[:5], key="drill_cats")
            if selected_cats:
                ddf = drill_df[drill_df["category"].astype(str).isin(selected_cats)]
                fig = plotly_line(ddf, "period", "revenue", "Revenue Over Time by Category", color="category")
                st.plotly_chart(fig, use_container_width=True)

        # Revenue vs Profit scatter (if both exist)
        rev_col  = det.get("revenue_col")
        prof_col = det.get("profit_col")
        if rev_col and prof_col and rev_col in filtered.columns and prof_col in filtered.columns:
            st.markdown("### 🔵 Revenue vs Profit")
            color_by = category_cols[0] if category_cols else None
            fig = px.scatter(filtered, x=rev_col, y=prof_col, color=color_by,
                             title="Revenue vs Profit", template=PLOTLY_TEMPLATE,
                             opacity=0.7, hover_data=filtered.columns[:5].tolist())
            st.plotly_chart(fig, use_container_width=True)

        # Summary tables
        if isinstance(breakdown_df, pd.DataFrame) and not breakdown_df.empty:
            with st.expander("📋 Revenue by Category (Table)"):
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
        if isinstance(profit_bdown_df, pd.DataFrame) and not profit_bdown_df.empty:
            with st.expander("📋 Profit by Category (Table)"):
                st.dataframe(profit_bdown_df, use_container_width=True, hide_index=True)

        # Executive summary
        st.divider()
        st.markdown("### 📝 Executive Summary")
        prompt = dk.build_exec_summary_prompt(spec)
        summary = llm_callable(prompt) if llm_callable else None
        st.markdown(summary or "_Connect LLM to generate a narrative summary._")

    # ==========================
    # CLASSIFICATION DASHBOARD
    # ==========================
    elif dtype == "classification":
        cb   = spec.get("class_balance")
        sep  = spec.get("top_separation")
        corr = spec.get("corr_pairs")
        nums = spec.get("numeric_summary")

        col_l, col_r = st.columns(2)

        if isinstance(cb, pd.DataFrame) and not cb.empty:
            with col_l:
                fig = plotly_pie(cb, "class", "count", "🎯 Class Balance")
                st.plotly_chart(fig, use_container_width=True)

        if isinstance(sep, pd.DataFrame) and not sep.empty:
            val_col = sep.columns[1]
            with col_r:
                fig = plotly_bar(sep.head(10), val_col, "feature",
                                 "🔬 Top Differentiating Features", orientation="h")
                st.plotly_chart(fig, use_container_width=True)

        if isinstance(corr, pd.DataFrame) and not corr.empty and len(corr) >= 3:
            fig = plotly_corr_heatmap(corr)
            st.plotly_chart(fig, use_container_width=True)

        if isinstance(nums, pd.DataFrame) and not nums.empty:
            st.markdown("### 📊 Numeric Feature Summary")
            st.dataframe(nums, use_container_width=True, hide_index=True)

    # ==========================
    # GENERIC DASHBOARD
    # ==========================
    else:
        corr = spec.get("corr_pairs")
        nums = spec.get("numeric_summary")
        cats = spec.get("categorical_summaries", [])

        if len(num_cols) < 2 and not cats:
            st.info("ℹ️ Not enough structured columns to generate meaningful charts. "
                    "Try the **Ask AI** tab for custom analysis.")
            return

        if isinstance(corr, pd.DataFrame) and not corr.empty and len(corr) >= 3:
            fig = plotly_corr_heatmap(corr)
            st.plotly_chart(fig, use_container_width=True)

        if num_cols:
            st.markdown("### 📊 Numeric Distributions")
            sel = st.selectbox("Select column for distribution", num_cols[:20], key="dist_col")
            fig = px.histogram(df, x=sel, nbins=40, title=f"Distribution of {sel}",
                               template=PLOTLY_TEMPLATE, marginal="box")
            st.plotly_chart(fig, use_container_width=True)

        for block in cats:
            c_name = block["col"]
            vc = block["top_counts"]
            fig = plotly_bar(vc, "count", c_name, f"Top Values: {c_name}", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

        if isinstance(nums, pd.DataFrame) and not nums.empty:
            st.markdown("### 📋 Numeric Summary Table")
            st.dataframe(nums, use_container_width=True, hide_index=True)

    # Export spec JSON
    with st.expander("⚙️ Dashboard Spec (JSON)", expanded=False):
        st.download_button(
            label="⬇️ Download Dashboard Spec",
            data=json.dumps(spec, default=str, indent=2).encode(),
            file_name="dashboard_spec.json",
            mime="application/json",
            key="dl_dashboard_spec",
        )


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    for k, v in [("show_code", False), ("show_meta", False),
                 ("generate_insight_toggle", True)]:
        if k not in st.session_state:
            st.session_state[k] = v

    with st.expander("AI Output Settings", expanded=False):
        st.session_state.generate_insight_toggle = st.toggle(
            "Generate AI insights", value=st.session_state.generate_insight_toggle)
        st.session_state.show_code = st.toggle(
            "Show generated code", value=st.session_state.show_code)
        st.session_state.show_meta = st.toggle(
            "Show complexity + bias/risk", value=st.session_state.show_meta)

    st.divider()
    st.subheader("🧾 History")
    st.write(f"Saved runs: **{len(st.session_state.history)}**")
    if st.button("Clear history"):
        st.session_state.history = []
        st.success("History cleared.")

show_code              = st.session_state.show_code
show_meta              = st.session_state.show_meta
generate_insight_toggle = st.session_state.generate_insight_toggle




# ----------------------------
# File upload
# ----------------------------

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center">
    Built by <b>Sai Tarun Reddy</b><br>
    <a href="https://github.com/tarunn31/ai-data-copilot">GitHub Repository</a>
    </div>
     """,
    unsafe_allow_html=True
)

sheet = None
dataset_label = None
df = None

if uploaded:
    dataset_label = uploaded.name
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        uploaded.seek(0)
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox("Select sheet", xls.sheet_names)
        dataset_label = f"{uploaded.name} / {sheet}"

    df = load_table(uploaded, sheet_name=sheet)
    raw_cols = [str(c).strip() for c in df.columns]
    dupes = pd.Index(raw_cols)[pd.Index(raw_cols).duplicated()].tolist()
    df = make_unique_columns(df)
    if dupes:
        st.warning(f"Duplicate column names auto-renamed: {sorted(set(dupes))}")

    # ----------------------------
    # TABS
    # ----------------------------
    tab_overview, tab_quality, tab_kpi, tab_askai, tab_history = st.tabs(
        ["📁 Overview & Preview", "🧼 Data Quality", "📊 KPI Dashboard", "💬 Ask AI", "🧾 History"]
    )

    # ========== TAB 1: Overview + Preview ==========
    with tab_overview:
        st.subheader("Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

        st.divider()
        st.subheader("Data Preview")
        n_rows = st.slider("Rows to display", 5, min(200, len(df)), 10, key="preview_slider")
        st.dataframe(df.head(n_rows), use_container_width=True)

        st.divider()
        st.subheader("Column Info")
        col_info = pd.DataFrame([{
            "Column": c,
            "Type": str(df[c].dtype),
            "Non-null": int(df[c].notna().sum()),
            "Missing %": f"{df[c].isna().mean()*100:.1f}%",
            "Unique values": int(df[c].nunique(dropna=True)),
            "Sample": str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else "—",
        } for c in df.columns])
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    # ========== TAB 2: Data Quality ==========
    with tab_quality:
        st.subheader("🧼 AI-Powered Data Quality Report")

        # Always show hard metrics
        rows, cols_n = df.shape
        dup_count  = int(df.duplicated().sum())
        miss_total = int(df.isna().sum().sum())
        miss_pct   = (df.isna().sum() / rows * 100).round(2)
        top_miss   = miss_pct[miss_pct > 0].sort_values(ascending=False).head(8)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Rows",           f"{rows:,}")
        mc2.metric("Columns",        f"{cols_n:,}")
        mc3.metric("Missing Cells",  f"{miss_total:,}")
        mc4.metric("Duplicate Rows", f"{dup_count:,}")

        if not top_miss.empty:
            st.markdown("**Top columns with missing values:**")
            miss_df = pd.DataFrame({"Column": top_miss.index, "Missing %": top_miss.values})
            fig = plotly_bar(miss_df, "Missing %", "Column",
                             "Missing Values by Column (%)", orientation="h")
            fig.update_layout(height=max(250, len(top_miss) * 35))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### 🤖 AI Narrative Analysis")

        if "quality_report" not in st.session_state or st.session_state.get("quality_dataset") != dataset_label:
            if st.button("Generate AI Quality Report", type="primary"):
                with st.spinner("Analysing data quality with AI..."):
                    report = generate_ai_quality_report(df)
                    st.session_state.quality_report   = report
                    st.session_state.quality_dataset  = dataset_label
                st.rerun()
        else:
            st.markdown(st.session_state.quality_report)
            if st.button("Regenerate Report"):
                del st.session_state["quality_report"]
                st.rerun()

    # ========== TAB 3: KPI Dashboard ==========
    with tab_kpi:
        st.subheader("📊 Interactive KPI Dashboard")
        render_kpi_dashboard(df, llm_callable=llm_exec_summary)

    # ========== TAB 4: Ask AI ==========
    with tab_askai:
        st.subheader("💬 Ask Anything About Your Data")
        question = st.text_input("Ask a question (analytical or exploratory)",
                                 placeholder="e.g. What is the average revenue by category? / What does the Sales column represent?")

        col_a, col_b = st.columns([1, 3])
        with col_a:
            run_btn = st.button("Run", type="primary")
        with col_b:
            st.caption("Can ask for tables, charts, summaries, or plain explanations.")

        if run_btn:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        analytical = is_analytical_question(question, df)

                        if analytical:
                            result_df, result_fig, code, attempts, agent_err = agent_run(df, question, max_retries=3)

                            if agent_err:
                                # Code failed — fall back to text answer
                                text_answer = answer_text_question(question, df)
                                st.session_state.last_run = {
                                    **st.session_state.last_run,
                                    "has_result": True,
                                    "dataset_label": dataset_label,
                                    "question": question,
                                    "code": None,
                                    "result_df": None,
                                    "result_fig": None,
                                    "insights": None,
                                    "attempts": attempts,
                                    "text_answer": text_answer,
                                    "report_md": None,
                                }
                            else:
                                insights_text = None
                                if generate_insight_toggle and isinstance(result_df, pd.DataFrame):
                                    insights_text = generate_insights(question, result_df)

                                report_md = build_markdown_report(
                                    dataset_name=dataset_label, question=question,
                                    code=code, result_df=result_df, insights=insights_text)

                                st.session_state.last_run = {
                                    "has_result": True,
                                    "dataset_label": dataset_label,
                                    "question": question,
                                    "code": code,
                                    "result_df": result_df,
                                    "result_fig": result_fig,
                                    "meta": None,
                                    "insights": insights_text,
                                    "report_md": report_md,
                                    "attempts": attempts,
                                    "text_answer": None,
                                }
                                st.session_state.history.append({
                                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "question": question,
                                    "code": code,
                                    "insights": insights_text,
                                    "result_preview": result_df.head(20).copy() if isinstance(result_df, pd.DataFrame) else None,
                                    "dataset": dataset_label,
                                })
                        else:
                            # Plain text answer grounded in dataset context
                            text_answer = answer_text_question(question, df)
                            st.session_state.last_run = {
                                **st.session_state.last_run,
                                "has_result": True,
                                "dataset_label": dataset_label,
                                "question": question,
                                "code": None,
                                "result_df": None,
                                "result_fig": None,
                                "insights": None,
                                "attempts": None,
                                "text_answer": text_answer,
                                "report_md": None,
                            }

                    except Exception as e:
                        st.error(f"Error: {e}")

        # ---- Render persisted results ----
        lr = st.session_state.last_run
        if lr.get("has_result") and lr.get("dataset_label") == dataset_label:
            st.divider()
            st.caption(f"**Q:** {lr.get('question', '')}")

            # Text answer (non-analytical or fallback)
            if lr.get("text_answer"):
                st.markdown("### 💬 Answer")
                st.markdown(lr["text_answer"])

            if lr.get("attempts") and show_code:
                with st.expander("🛠 Agent Attempts"):
                    st.json(lr["attempts"])

            if show_code and lr.get("code"):
                with st.expander("🧠 Generated Code"):
                    st.code(lr["code"], language="python")

            if lr.get("result_df") is not None:
                st.markdown("### 📋 Result Table")
                st.dataframe(lr["result_df"], use_container_width=True)

            if lr.get("result_fig") is not None:
                st.markdown("### 📈 Chart")
                st.pyplot(lr["result_fig"], clear_figure=True)

            if generate_insight_toggle and lr.get("insights"):
                st.markdown("### 🧠 AI Insights")
                st.markdown(lr["insights"])

            if lr.get("report_md"):
                st.download_button(
                    label="⬇️ Download Report (Markdown)",
                    data=lr["report_md"].encode(),
                    file_name="ai_copilot_report.md",
                    mime="text/markdown",
                    key="dl_report",
                )

    # ========== TAB 5: History ==========
    with tab_history:
        st.subheader("🧾 Run History (latest first)")
        if not st.session_state.history:
            st.info("No runs yet. Use Ask AI to get started.")
        else:
            for item in reversed(st.session_state.history[-20:]):
                with st.expander(f"{item['time']} — {item['question']}"):
                    st.write(f"**Dataset:** {item['dataset']}")
                    if show_code and item.get("code"):
                        st.code(item["code"], language="python")
                    if item.get("result_preview") is not None:
                        st.dataframe(item["result_preview"], use_container_width=True)
                    if item.get("insights"):
                        st.markdown("**Insights:**")
                        st.markdown(item["insights"])

else:
    st.info("👆 Upload a CSV or Excel file to get started.")


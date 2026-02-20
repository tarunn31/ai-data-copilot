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
"""



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
    local_env = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "result_df": None,
        "result_fig": None,
    }
    exec(code, {"__builtins__": {}}, local_env)
    return local_env.get("result_df"), local_env.get("result_fig")


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

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    show_code = st.toggle("Show generated code", value=True)
    show_meta = st.toggle("Show complexity + bias/risk", value=True)
    generate_insight_toggle = st.toggle("Generate AI insights", value=True)

    st.divider()
    st.subheader("🧾 History")
    st.write(f"Saved runs: **{len(st.session_state.history)}**")
    if st.button("Clear history"):
        st.session_state.history = []
        st.success("History cleared.")

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

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()
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
                with st.spinner("Generating analysis code..."):
                    code = generate_pandas_code(df, question)

                # Remove harmless import lines (we already provide pd/np/plt)
                code = re.sub(r"^import .*$", "", code, flags=re.MULTILINE)
                code = re.sub(r"^from .*$", "", code, flags=re.MULTILINE)
                code = code.strip()

                ok, reason = is_code_safe(code)
                if not ok:
                    st.error(f"Generated code blocked for safety. {reason}")
                else:
                    with st.spinner("Running analysis..."):
                        result_df, result_fig = run_generated_code(df, code)

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

                    # ✅ Persist results for reruns (toggles won't clear output)
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

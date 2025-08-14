from __future__ import annotations
import os, datetime, json
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env into environment for os.getenv(...)
except Exception:
    pass

# ---- Load CENSUS_API_KEY from .env / secrets (robust) ----
# ------- TOP-OF-FILE BOOTSTRAP (put this above any st.* usage) -------
import os
from pathlib import Path

# Load .env so os.getenv("CENSUS_API_KEY") works (no Streamlit calls here)
try:
    from dotenv import load_dotenv, find_dotenv
    # 1) load .env next to this file if present
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
    # 2) also search upwards from current working dir
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
except Exception:
    pass

import streamlit as st  # import AFTER dotenv
# Make this the FIRST Streamlit call in the script, and only call it ONCE.
st.set_page_config(page_title="Oil & Trade for India", layout="wide")
# ---------------------------------------------------------------------
import os
# Show whether the key was picked up
_key = os.getenv("CENSUS_API_KEY", "")

st.sidebar.caption(f"Census API key: {'✅ detected' if _key else '❌ not found'}")
st.sidebar.text(f"Masked: {'…' + _key[-4:] if _key else '(none)'}")

# Optional: reveal full key only if you explicitly choose to (unsafe to keep enabled)
if _key and st.sidebar.checkbox("Show FULL API key (unsafe)"):
    st.sidebar.code(_key)

# -----------------------------------------------------------

# Optional deps
try:
    import requests
except Exception:
    requests = None

try:
    import altair as alt
except Exception:
    alt = None

try:
    import pandas as pd
except Exception:
    pd = None

# Backend module (same folder)
load_dotenv()
import backend_oil_import_optimizerandTrade as be

# --------------------------------------------------------------------------------------
# Page & styling
# --------------------------------------------------------------------------------------

# Single-line, horizontally scrollable tab bar
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] { flex-wrap: nowrap; overflow-x: auto; row-gap: 0; scrollbar-width: thin; }
    .stTabs [data-baseweb="tab"] { white-space: nowrap; padding-top: .25rem; padding-bottom: .25rem; }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { height: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Auto refresh (ms)
auto_ms = int(os.getenv("AUTO_REFRESH_MS", "60000"))
st_autorefresh(interval=auto_ms, key="data_refresh")



# --------------------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------------------
st.sidebar.title("Scenario Controls (Live)")

# Oil-side knobs
_tariff_rate = st.sidebar.slider("Tariff shock (crude surcharge proxy)", 0.0, 0.25, 0.10, 0.01, format="%.2f")
_demand_bpd = st.sidebar.number_input("Daily import demand (bbl/day)", 200_000, 10_000_000, 4_500_000, 50_000)
_max_supplier_share = st.sidebar.slider("Max share per supplier", 0.10, 1.00, 0.45, 0.05)
_min_non_russia_share = st.sidebar.slider("Minimum non-Russia share", 0.0, 1.0, 0.55, 0.05)
_selected_port = st.sidebar.selectbox("Destination port (India)", ["India-Mundra", "India-JNPT", "India-Paradip"])
_objective = st.sidebar.selectbox("Optimizer objective", ["hybrid", "cost", "risk"])

st.sidebar.divider()

# Russia discount & intake
_rus_discount_pct = st.sidebar.slider("Russian Oil Discount (% of Brent)", 0.0, 10.0, 4.0, 0.5)
_rus_bpd = st.sidebar.number_input("Russia crude intake (bbl/day)", 0, 5_000_000, 1_800_000, 50_000)

# U.S. tariffs & diversion modeling
_us_tariff_pct = st.sidebar.select_slider("U.S. tariff level on India (goods)", options=[0, 10, 15, 25, 35, 50], value=25)
_elasticity = st.sidebar.slider("U.S. import demand elasticity", 0.5, 2.5, 1.5, 0.1)

_divert_pct = st.sidebar.slider("Divertable to BRICS (% of U.S.-bound)", 0.0, 100.0, 40.0, 5.0)
_markdown_pct = st.sidebar.slider("Markdown to sell in BRICS (%)", 0.0, 25.0, 8.0, 1.0)
_extra_cost_pct = st.sidebar.slider("Extra go-to-market cost in BRICS (%)", 0.0, 10.0, 2.0, 0.5)

st.sidebar.caption("Live Brent via yfinance; Dubai/Urals fallback in backend. U.S. Census API for U.S. imports from India.")

# --------------------------------------------------------------------------------------
# Backend call (graceful fallback if backend lacks russia_discount_pct)
# --------------------------------------------------------------------------------------
try:
    _result = be.run_full_pipeline(
        tariff_rate=_tariff_rate,
        demand_bpd=_demand_bpd,
        max_supplier_share=_max_supplier_share,
        min_non_russia_share=_min_non_russia_share,
        selected_indian_port=_selected_port,
        optimizer_objective=_objective,
        russia_discount_pct=_rus_discount_pct,
    )
    _backend_supports_rus_discount = True
except TypeError:
    _result = be.run_full_pipeline(
        tariff_rate=_tariff_rate,
        demand_bpd=_demand_bpd,
        max_supplier_share=_max_supplier_share,
        min_non_russia_share=_min_non_russia_share,
        selected_indian_port=_selected_port,
        optimizer_objective=_objective,
    )
    _backend_supports_rus_discount = False
except Exception as e:
    st.error(f"Backend error: {e}")
    st.stop()

_sc = _result["scenario"]
_quotes = _result["quotes"]
_mix = _result["mix"]
_metrics = _result["metrics"]
_bench = _result["live_benchmarks"]
_sentiment = _result["news_sentiment"]
_headlines = _result["headlines"]
_advisory = _result["advisory"]
_used_llm = _result["advisory_used_llm"]

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _brent_price() -> float:
    try:
        return float(_bench.get("Brent", 82.0))
    except Exception:
        return 82.0


def _fetch_us_imports_from_india_monthly(limit_months: int = 24):
    if requests is None:
        return [], None
    try:
        url = (
            "https://api.census.gov/data/timeseries/intltrade/imports/ctry"
            "?get=CTY_NAME,CTY_CODE,ALL_VAL_MO&time=from+2024-01&CTY_CODE=5330"
        )
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return [], None
        data = r.json()
        rows_in = data[1:]
        rows = []
        for cty_name, cty_code, all_val_mo, time_str in rows_in:
            try:
                val = float(all_val_mo)
            except Exception:
                continue
            rows.append({
                "time": time_str,
                "us_imports_from_india_usd": val,
                "us_imports_from_india_usd_billion": val/1e9,
            })
        rows = sorted(rows, key=lambda x: x["time"])[:limit_months]
        if not rows:
            return [], None
        last12 = [x["us_imports_from_india_usd"] for x in rows[-12:]]
        annual_usd = sum(last12)
        return rows, annual_usd
    except Exception:
        return [], None

_monthly_us_imports_from_india, _annual_us_imports_usd = _fetch_us_imports_from_india_monthly()

_brent = _brent_price()
_rus_discount_usd = _brent * (_rus_discount_pct/100.0)
_rus_month_bbl = _rus_bpd * 30
_monthly_russia_saving_usd = _rus_month_bbl * _rus_discount_usd


def _tariff_loss(annual_x_usd: float, tariff_pct: float, eps: float) -> float:
    t = tariff_pct/100.0
    return annual_x_usd * (1 - (1 + t)**(-eps))


def _diversion_gain(annual_x_usd: float, divert_pct: float, markdown_pct: float, extra_cost_pct: float) -> float:
    d = divert_pct/100.0
    m = markdown_pct/100.0
    c = extra_cost_pct/100.0
    return annual_x_usd * d * max(0.0, 1 - m - c)

# --------------------------------------------------------------------------------------
# Tabs — SINGLE ROW
# --------------------------------------------------------------------------------------
base_tabs = [
    "Overview",
    "Inputs & Assumptions",
    "Tariff & Landed Cost",
    "Supplier Mix Optimizer",
    "Geopolitical Risk (Live Feeds)",
    "Logistics Route & Time",
    "Sustainability & ESG",
    "GenAI Trade Advisor",
]

trade_tabs = [
    "BRICS Import Markets",
    "What to Move (Opportunities)",
    "Cheap Oil Link (Net)",
    "Diversion Calculator",
    "Guardrails & Compliance",
    "BRICS Diversion Planner",
]

all_tabs = base_tabs + trade_tabs
_tabs = st.tabs(all_tabs)
IDX = {name: i for i, name in enumerate(all_tabs)}

# --------------------------------------------------------------------------------------
# Content — Oil
# --------------------------------------------------------------------------------------
with _tabs[IDX["Overview"]]:
    st.markdown("## Oil & Trade for India (Layman Overview)")
    st.write(
        """
        **Goal:** Buy oil smartly at low cost while protecting India’s exports.

        **In simple words:**
        - **Live prices** for Brent/Dubai/Urals guide crude buying.
        - **AI optimizer** spreads purchases across countries to balance **cost, risk, and carbon**.
        - **Trade module** shows how much India sells to the **U.S.** and how much could be **shifted to BRICS** if U.S. tariffs rise.
        - You can set a **Russian discount (%)**, **Russia barrels/day**, and **U.S. tariff level** and see the **net effect**.
        """
    )

    status = "✅ OpenAI advisory enabled" if _used_llm else "⌛ Offline template (no OPENAI_API_KEY or API error)"
    st.info(f"Live status — Benchmarks (USD/bbl): {_bench} | Sentiment: {_sentiment:+.2f} | {status}")

    if not _backend_supports_rus_discount:
        st.warning("Backend not yet patched for 'Russian Oil Discount %' — optimizer is ignoring the discount slider.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Brent (live/fallback)", f"${_brent:.2f}/bbl")
    c2.metric("Russian Discount (setting)", f"{_rus_discount_pct:.1f}% → ${_rus_discount_usd:.2f}/bbl")
    c3.metric("Russia Intake", f"{_rus_bpd/1_000_000:.2f} million bbl/day")

    st.markdown("### Key Quantities (per month)")
    c4, c5 = st.columns(2)
    c4.metric("Russia barrels (per month)", f"{_rus_month_bbl:,.0f} bbl")
    c5.metric("Saving from Russia discount", f"${_monthly_russia_saving_usd/1e6:,.1f} M")

    if _annual_us_imports_usd:
        st.markdown("### U.S. Goods Imports from India (Census, trailing 12m)")
        st.metric("U.S. imports from India (annualized)", f"${_annual_us_imports_usd/1e9:,.2f} B")
        if alt and pd and _monthly_us_imports_from_india:
            df = pd.DataFrame(_monthly_us_imports_from_india)
            chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X("time:N", title="Month"),
                y=alt.Y("us_imports_from_india_usd_billion:Q", title="USD (B)"),
                tooltip=["time", "us_imports_from_india_usd_billion"],
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Could not fetch U.S. Census data live. Using your manual settings and optimizer only.")

    st.caption("Units: bbl = barrels. M = million. B = billion. U.S. Census values are goods only, USD.")

with _tabs[IDX["Inputs & Assumptions"]]:
    st.subheader("Live Benchmarks & Supplier Adjustments")
    st.json({"Live Benchmarks (USD/bbl)": _bench})
    rows = []
    for s in be.SUPPLIERS:
        bmk = be.SUPPLIER_BENCHMARK[s]
        base = _bench.get(bmk, None)
        base = round(float(base), 2) if base is not None else None
        adj = be.SUPPLIER_PRICE_ADJ[s]
        rows.append({"Supplier": s, "Benchmark": bmk, "Benchmark Price": base, "Supplier Adj (USD/bbl)": adj})
    st.dataframe(rows, use_container_width=True)

    st.write("**Freight & Other Assumptions** (per barrel)")
    st.json({
        "Freight Base": be.DEFAULT_FREIGHT_BASE,
        "Freight per 1000nm": be.DEFAULT_FREIGHT_PER_1000NM,
        "Handling": be.DEFAULT_HANDLING,
        "Insurance Rate": be.DEFAULT_INSURANCE_RATE,
        "Destination Port": _selected_port,
    }, expanded=False)

with _tabs[IDX["Tariff & Landed Cost"]]:
    st.subheader("Per-Supplier Landed Cost (USD/bbl)")
    rows = []
    for s, q in _quotes.items():
        rows.append({
            "Supplier": s,
            "Benchmark": q.benchmark,
            "Raw Price": round(q.raw_price, 2),
            "Tariff": round(q.tariff_component, 2),
            "Freight": round(q.freight, 2),
            "Handling": round(q.handling, 2),
            "Insurance": round(q.insurance, 2),
            "Landed Cost": round(q.landed_cost, 2),
        })
    st.dataframe(rows, use_container_width=True)
    st.caption("Landed = Raw + Tariff + Freight + Handling + Insurance")

with _tabs[IDX["Supplier Mix Optimizer"]]:
    st.subheader("Optimized Supplier Mix")
    mix_rows = []
    for s in be.SUPPLIERS:
        q = _quotes[s]
        mix_rows.append({
            "Supplier": s,
            "Share %": round(100 * _mix.get(s, 0.0), 1),
            "Landed (USD/bbl)": round(q.landed_cost, 2),
            "Carbon (kgCO2e/bbl)": q.carbon_intensity,
            "ESG": q.esg,
        })
    st.dataframe(mix_rows, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Weighted Landed Cost", f"${_metrics['weighted_landed_cost_usd_bbl']:.2f}/bbl")
    c2.metric("Weighted Carbon", f"{_metrics['weighted_carbon_kgco2e_bbl']:.1f} kgCO2e/bbl")
    c3.metric("Weighted ESG", f"{_metrics['weighted_esg']:.0f}")
    st.caption("Constraints: sum=100%, share≤cap, Non-Russia≥threshold (Russia forced = 1−Non-Russia). 5% grid.")

with _tabs[IDX["Geopolitical Risk (Live Feeds)"]]:
    st.subheader("Headline Sentiment (RSS)")
    st.write(f"**Avg Sentiment:** {_sentiment:+.2f} (−1..+1). Adjusted Stability = ESG + 10×Sentiment.")
    if _headlines:
        st.write("Latest headlines (sample):")
        st.write("\n".join([f"- {h}" for h in _headlines[:10]]))
    else:
        st.warning("No headlines fetched (offline or feed blocked). Using neutral sentiment.")

    adj_rows = []
    for s in be.SUPPLIERS:
        base = be.ESG_SCORE[s]
        adj = be.risk_score_from_sentiment(base, _sentiment)
        adj_rows.append({"Supplier": s, "Base ESG": base, "Adjusted Stability": round(adj, 1)})
    st.dataframe(adj_rows, use_container_width=True)

with _tabs[IDX["Logistics Route & Time"]]:
    st.subheader("Distance & Sailing Time")
    _speed_knots = st.slider("Assumed vessel speed (knots)", 10.0, 20.0, 14.0, 0.5)
    rows = []
    for s in be.SUPPLIERS:
        d_nm, days = be.shipping_time_days(s, _selected_port, _speed_knots)
        rows.append({
            "Supplier": s,
            "Route": f"{be.supplier_port(s)} → {_selected_port}",
            "Distance (nm)": int(d_nm),
            "Time (days)": round(days, 1),
        })
    st.dataframe(rows, use_container_width=True)
    st.caption("Time = Distance / (knots × 24).")

with _tabs[IDX["Sustainability & ESG"]]:
    st.subheader("Sustainability Overview of the Chosen Mix")
    st.markdown(f"- **Weighted Carbon**: **{_metrics['weighted_carbon_kgco2e_bbl']:.1f} kgCO2e/bbl**")
    st.markdown(f"- **Weighted ESG**: **{_metrics['weighted_esg']:.0f}**")
    rows = []
    for s in be.SUPPLIERS:
        rows.append({
            "Supplier": s,
            "Share %": round(100 * _mix.get(s, 0.0), 1),
            "Carbon": be.CARBON_INTENSITY[s],
            "ESG": be.ESG_SCORE[s],
        })
    st.dataframe(rows, use_container_width=True)
    st.caption("Weighted metrics = Σ Share_i × Metric_i")

with _tabs[IDX["GenAI Trade Advisor"]]:
    st.subheader("GenAI Policy Brief")
    if _used_llm:
        st.success("Generated using OpenAI (set OPENAI_API_KEY / OPENAI_MODEL).")
    else:
        st.info("Offline template (enable OPENAI_API_KEY for live GenAI).")
    st.write(_advisory)

# --------------------------------------------------------------------------------------
# Content — Trade/BRICS
# --------------------------------------------------------------------------------------
with _tabs[IDX["BRICS Import Markets"]]:
    st.subheader("BRICS / BRICS+ Import Markets (high-level)")
    st.write("Editable placeholders unless you wire an API.")
    defaults = {
        "China": 2580.0, "UAE": 539.0, "Brazil": 278.0, "Saudi Arabia": 250.0,
        "South Africa": 101.0, "Egypt": 95.0, "Russia": 270.0, "Iran": 35.0, "Ethiopia": 23.0,
    }
    markets = {}
    cols = st.columns(3)
    for i, k in enumerate(defaults.keys()):
        with cols[i % 3]:
            markets[k] = st.number_input(f"{k} imports (USD B, last yr)", 0.0, 5000.0, float(defaults[k]), 1.0)

    # Save for use in the Opportunities tab
    st.session_state["brics_markets"] = markets
    st.session_state["brics_markets_total_usd_b"] = float(sum(markets.values()))

    if alt and pd:
        df = pd.DataFrame([{"Market": k, "Imports_USD_B": v} for k, v in markets.items()])
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Market:N", sort="-y"),
            y=alt.Y("Imports_USD_B:Q", title="Imports (USD B)"),
            tooltip=["Market", "Imports_USD_B"],
        )
        st.altair_chart(chart, use_container_width=True)

with _tabs[IDX["What to Move (Opportunities)"]]:
    # ---------------- Imports (local to this tab) ----------------
    import os, time
    import requests
    import pandas as pd
    import streamlit as st
    try:
        import altair as alt
    except Exception:
        alt = None

    # ---------------- Header ----------------
    st.subheader("Quantified opportunities: U.S. baseline → BRICS potential (net of BRICS self-supply) — Product-wise (HS2)")

    # ---------------- Pull global controls (fallbacks if not defined) ----------------
    def _fallback(var_name, default):
        try:
            return globals()[var_name]
        except Exception:
            return st.session_state.get(var_name, default)

    _divert_pct     = float(_fallback("_divert_pct", 25.0))    # %
    _markdown_pct   = float(_fallback("_markdown_pct", 5.0))   # %
    _extra_cost_pct = float(_fallback("_extra_cost_pct", 3.0)) # %

    # ---------------- Tiny per-session cache ----------------
    if "unct_cache" not in st.session_state:
        st.session_state["unct_cache"] = {}
    def _cache_get(key):
        it = st.session_state["unct_cache"].get(key)
        if not it: return None
        ts, ttl, data = it
        return data if (time.time() - ts) <= ttl else None
    def _cache_set(key, data, ttl=3600):
        st.session_state["unct_cache"][key] = (time.time(), ttl, data)

    # ---------------- BRICS headroom (net of intra-supply) ----------------
    _brics_defaults = {
        "China": 2580.0, "UAE": 539.0, "Brazil": 278.0, "Saudi Arabia": 250.0,
        "South Africa": 101.0, "Egypt": 95.0, "Russia": 270.0, "Iran": 35.0, "Ethiopia": 23.0,
    }
    brics_markets = st.session_state.get("brics_markets", _brics_defaults)
    brics_total_imports_usd = float(sum(brics_markets.values())) * 1e9

    intra_share = st.slider(
        "BRICS intra-supply share (already supplied within BRICS, %)",
        min_value=0, max_value=100, value=60, step=5, key="opp_intra_share"
    )
    brics_self_supply_usd = brics_total_imports_usd * (intra_share / 100.0)
    opportunity_pool_usd = max(0.0, brics_total_imports_usd - brics_self_supply_usd)

    # ---------------- Data source + year ----------------
    c1, c2 = st.columns([1,1])
    with c1:
        year = st.number_input("Trade year (data source below)", min_value=2000, max_value=2100, value=2023, step=1, key="opp_year")
    with c2:
        source = st.radio("Data source", ["Census API (live)", "Upload CSV"], index=0, horizontal=True)

    hs_level = 2  # HS2 only

    # ---------------- Robust Census fetch (fast-fail + year fallback) ----------------
    def fetch_census_hs2(year_: int) -> pd.DataFrame:
        """
        Fast-fail fetch of U.S. imports from India (HS2) using U.S. Census API.
        Tries timeseries Dec (annual YTD) then annual dataset. Returns product_code, product_name, us_baseline_usd.
        """
        cache_key = f"census_hs2:{year_}:IN"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        def _is_json(resp):
            return resp.headers.get("Content-Type","").lower().startswith("application/json")

        def _build_df(rows, header):
            df = pd.DataFrame(rows, columns=header)
            if df.empty:
                return pd.DataFrame(columns=["product_code","product_name","us_baseline_usd"])
            # Choose fields (timeseries/annual typically use these names)
            code_col = "I_COMMODITY" if "I_COMMODITY" in df.columns else ("HS_CODE" if "HS_CODE" in df.columns else None)
            name_col = "I_COMMODITY_LDESC" if "I_COMMODITY_LDESC" in df.columns else ("HS_DESC" if "HS_DESC" in df.columns else (code_col or "I_COMMODITY"))
            val_col  = "GEN_VAL_YR" if "GEN_VAL_YR" in df.columns else ("GEN_VAL_MO" if "GEN_VAL_MO" in df.columns else None)
            if code_col is None or val_col is None:
                return pd.DataFrame(columns=["product_code","product_name","us_baseline_usd"])
            df["product_code"] = df[code_col].astype(str).str.slice(0, hs_level)
            df["product_name"] = df[name_col].astype(str)
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
            out = (df.groupby(["product_code","product_name"], as_index=False)[val_col]
                     .sum()
                     .rename(columns={val_col:"us_baseline_usd"})
                     .sort_values("us_baseline_usd", ascending=False))
            return out

        api_key = os.getenv("CENSUS_API_KEY","")
        tries = [
            # Timeseries December (annual YTD)
            ("https://api.census.gov/data/timeseries/intltrade/imports/hs",
             {"get":"CTY_CODE,CTY_NAME,COMM_LVL,I_COMMODITY,I_COMMODITY_LDESC,GEN_VAL_YR,GEN_VAL_MO",
              "time": f"{year_}-12", "CTY_CODE":"IN", "COMM_LVL":"HS2"}),
            # Annual dataset
            (f"https://api.census.gov/data/{year_}/intltrade/imports/hs",
             {"get":"CTY_CODE,CTY_NAME,COMM_LVL,I_COMMODITY,I_COMMODITY_LDESC,GEN_VAL_YR",
              "CTY_CODE":"IN", "COMM_LVL":"HS2"}),
        ]
        if api_key:
            for _, p in tries: p["key"] = api_key

        OVERALL_DEADLINE = time.monotonic() + 12.0
        PER_REQ_TIMEOUT  = 5.0
        MAX_RETRIES      = 1

        for url, params in tries:
            attempt = 0
            while attempt <= MAX_RETRIES and time.monotonic() < OVERALL_DEADLINE:
                attempt += 1
                try:
                    resp = requests.get(url, params=params, timeout=PER_REQ_TIMEOUT)
                    if resp.status_code in (429,500,502,503,504):
                        time.sleep(0.7)
                        continue
                    resp.raise_for_status()
                    if not _is_json(resp): break
                    js = resp.json()
                    if not js or len(js) < 2: break
                    header, rows = js[0], js[1:]
                    out = _build_df(rows, header)
                    if not out.empty:
                        _cache_set(cache_key, out, ttl=3600)
                        return out
                    break
                except Exception:
                    # try next attempt/endpoint quickly
                    pass
        # empty if all attempts failed
        return pd.DataFrame(columns=["product_code","product_name","us_baseline_usd"])

    def fetch_census_hs2_with_year_fallback(selected_year: int) -> tuple[pd.DataFrame, int]:
        for yr in [selected_year, selected_year-1, selected_year-2]:
            if yr < 2000: break
            df_try = fetch_census_hs2(yr)
            if not df_try.empty:
                return df_try, yr
        return pd.DataFrame(columns=["product_code","product_name","us_baseline_usd"]), selected_year

    # ---------------- CSV upload parser (real-data fallback) ----------------
    def parse_uploaded_csv(file) -> pd.DataFrame:
        try:
            raw = pd.read_csv(file)
        except Exception:
            file.seek(0)
            raw = pd.read_excel(file)
        cols = {c.lower(): c for c in raw.columns}
        code_col = cols.get("hs2") or cols.get("hs_code") or cols.get("i_commodity") or cols.get("cmdcode") or list(raw.columns)[0]
        name_col = cols.get("description") or cols.get("hs_desc") or cols.get("i_commodity_ldesc") or cols.get("cmddesce") or list(raw.columns)[1]
        val_col  = cols.get("value") or cols.get("us_baseline_usd") or cols.get("gen_val_yr") or cols.get("tradevalue") or list(raw.columns)[-1]
        df = pd.DataFrame({
            "product_code": raw[code_col].astype(str).str.slice(0, hs_level),
            "product_name": raw[name_col].astype(str),
            "us_baseline_usd": pd.to_numeric(raw[val_col], errors="coerce").fillna(0.0)
        })
        return (df.groupby(["product_code","product_name"], as_index=False)["us_baseline_usd"]
                  .sum()
                  .sort_values("us_baseline_usd", ascending=False))

    # ---------------- Acquire product-wise data ----------------
    df = pd.DataFrame(columns=["product_code","product_name","us_baseline_usd"])
    used_year = year

    if source == "Census API (live)":
        with st.spinner("Loading product-wise trade data… (fast-fail + year fallback)"):
            df, used_year = fetch_census_hs2_with_year_fallback(year)
        if df.empty:
            st.warning("No product data returned from public endpoints right now. "
                       "Try another year, ensure CENSUS_API_KEY is set, or use 'Upload CSV'.")
    else:
        up = st.file_uploader("Upload a CSV/XLSX with product-wise U.S. imports from India (HS2/HS6).", type=["csv","xlsx"])
        st.caption("Expected columns (any order): HS2/HS_CODE/I_COMMODITY, Description, Value (USD). HS6 is OK; we aggregate to HS2.")
        if up is not None:
            with st.spinner("Parsing uploaded file…"):
                try:
                    df = parse_uploaded_csv(up)
                except Exception as e:
                    st.error(f"Could not parse file: {e}")
                    df = pd.DataFrame(columns=["product_code","product_name","us_baseline_usd"])

    # ---------------- Show raw baselines table ----------------
    st.markdown(f"### Product-wise baselines (HS2) — year used: **{used_year}**")
    st.dataframe(
        df.rename(columns={
            "product_code":"HS2",
            "product_name":"Description",
            "us_baseline_usd":"U.S. Baseline (USD)"
        }).sort_values("U.S. Baseline (USD)", ascending=False),
        use_container_width=True, height=320
    )

    # ---------------- Caps & summaries ----------------
    total_us_baseline = float(df["us_baseline_usd"].sum()) if not df.empty else 0.0
    india_divertable_usd = total_us_baseline * (_divert_pct / 100.0)
    india_to_brics_cap_usd = min(india_divertable_usd, opportunity_pool_usd)

    c1, c2, c3 = st.columns(3)
    c1.metric("India → U.S. (baseline, annualized)", f"${total_us_baseline/1e9:,.2f} B")
    c2.metric("BRICS intra-supply (not opportunity)", f"${brics_self_supply_usd/1e9:,.2f} B")
    c3.metric("India → BRICS (divertable cap)", f"${india_to_brics_cap_usd/1e9:,.2f} B")
    st.caption("India→BRICS potential = min( U.S.-bound × divert% , BRICS total imports − BRICS intra-supply ).")

    # ---------------- Per-product allocation & opportunity ----------------
    if not df.empty:
        df["weight"] = df["us_baseline_usd"] / max(1e-9, total_us_baseline)
        df["divertable_usd"] = df["us_baseline_usd"] * (_divert_pct / 100.0)
        df["brics_headroom_usd"] = opportunity_pool_usd * df["weight"]
        realization_factor = max(0.0, 1 - (_markdown_pct + _extra_cost_pct) / 100.0)
        df["realizable_usd"] = df[["divertable_usd","brics_headroom_usd"]].min(axis=1) * realization_factor

        view = df.copy()
        view["US_Baseline_USD_B"]     = view["us_baseline_usd"]    / 1e9
        view["Divertable_USD_B"]      = view["divertable_usd"]     / 1e9
        view["BRICS_Headroom_USD_B"]  = view["brics_headroom_usd"] / 1e9
        view["Realizable_USD_B"]      = view["realizable_usd"]     / 1e9

        st.markdown("### Product-wise (HS2) results")

        # Results table (show Pretty label if exists)
        cols = ["HS2","Description","US_Baseline_USD_B","Divertable_USD_B","BRICS_Headroom_USD_B","Realizable_USD_B"]
        tmp = view.rename(columns={"product_code":"HS2","product_name":"Description"}).copy()
        if "Pretty" in view.columns:
            tmp.insert(1, "Pretty", view["Pretty"])
            cols = ["HS2","Pretty","Description","US_Baseline_USD_B","Divertable_USD_B","BRICS_Headroom_USD_B","Realizable_USD_B"]

        st.dataframe(
            tmp[cols].sort_values("Realizable_USD_B", ascending=False),
            use_container_width=True, height=420
        )

        # Chart with Description (or Pretty) labels, HS2 in tooltip
        if alt:
            chart_df = view.rename(columns={"product_code":"HS2","product_name":"Description"}).copy()
            name_for_axis = "Pretty" if "Pretty" in chart_df.columns else "Description"
            st.altair_chart(
                alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X(f"{name_for_axis}:N", sort="-y", title="Product (HS2)", axis=alt.Axis(labelAngle=-30, labelLimit=280)),
                    y=alt.Y("Realizable_USD_B:Q", title="Realizable Opportunity (USD B)"),
                    tooltip=[name_for_axis, "HS2", "Realizable_USD_B"]
                ).properties(height=380),
                use_container_width=True
            )

        st.metric("Total Realizable (all products)", f"${view['Realizable_USD_B'].sum():,.2f} B")
    else:
        view = pd.DataFrame(columns=["product_code","product_name","Realizable_USD_B"])
        st.info("No product rows to compute opportunities. Use a different year or upload CSV to proceed.")

    # ---------------- GenAI (optional) ----------------
    import json
    from textwrap import dedent
    from functools import lru_cache

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _ai_available() -> bool:
        return bool(OPENAI_API_KEY)

    def _make_prompt(rows, globals_dict):
        return dedent(f"""
        You are a trade strategy analyst. Explain, in JSON, why each product ranks where it does.
        Use fields: reasons (3 short bullets), risks (2 short bullets), actions (2 short bullets).
        Keep each bullet <= 18 words. Be concrete and reference numeric drivers.

        GLOBALS:
        - divert_pct={globals_dict['_divert_pct']}%
        - markdown_pct={globals_dict['_markdown_pct']}%
        - extra_cost_pct={globals_dict['_extra_cost_pct']}%
        - intra_supply_share={globals_dict['intra_share']}%
        - total_headroom_USD_B={globals_dict['opportunity_pool_usd']/1e9:.2f}

        PRODUCTS (Top {len(rows)} by Realizable_USD_B):
        {json.dumps(rows, ensure_ascii=False)}
        """)

    def _ai_explain(rows, globals_dict):
        if not _ai_available():
            return {"error": "OPENAI_API_KEY missing"}
        try:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                rsp = client.chat.completions.create(
                    model=OPENAI_MODEL, temperature=0.2,
                    response_format={"type":"json_object"},
                    messages=[{"role":"system","content":"Return valid JSON only."},
                              {"role":"user","content":_make_prompt(rows, globals_dict)}]
                )
                content = rsp.choices[0].message.content
            except Exception:
                import openai
                openai.api_key = OPENAI_API_KEY
                rsp = openai.ChatCompletion.create(
                    model=OPENAI_MODEL, temperature=0.2,
                    messages=[{"role":"system","content":"Return valid JSON only."},
                              {"role":"user","content":_make_prompt(rows, globals_dict)}]
                )
                content = rsp["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    @lru_cache(maxsize=200)
    def ai_label_hs2(hs2: str, desc: str) -> str:
        if not _ai_available():
            return desc
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = f"Give a short business-friendly category for HS{hs2} ({desc}). <= 4 words."
            out = client.chat.completions.create(
                model=OPENAI_MODEL, temperature=0.2,
                messages=[{"role":"user","content":prompt}]
            )
            return out.choices[0].message.content.strip()
        except Exception:
            return desc

    def ai_scenario_narrative(metrics: dict) -> str:
        if not _ai_available():
            return ""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = f"""In ~120 words, explain this scenario for a CFO:\n{json.dumps(metrics, indent=2)}\nFocus on what changed and why. Avoid hype."""
            out = client.chat.completions.create(
                model=OPENAI_MODEL, temperature=0.3,
                messages=[{"role":"system","content":"Clear, factual, CFO tone."},
                          {"role":"user","content":prompt}]
            )
            return out.choices[0].message.content.strip()
        except Exception:
            return ""

    # ---- AI insights UI ----
    st.markdown("#### AI insights (optional)")
    n_top = st.slider("Explain top N products", 3, 15, 5, 1, key="ai_top_n")
    go_ai = st.button("Generate AI insights", disabled=(not _ai_available()) or view.empty)

    if not _ai_available():
        st.caption("⚠️ Set OPENAI_API_KEY in your .env to enable AI insights.")

    if go_ai and not view.empty:
        _tmp_for_ai = view.rename(columns={"product_code":"HS2","product_name":"Description"}).copy()
        top_rows = (_tmp_for_ai[["HS2","Description","US_Baseline_USD_B","Divertable_USD_B","BRICS_Headroom_USD_B","Realizable_USD_B"]]
                        .sort_values("Realizable_USD_B", ascending=False)
                        .head(n_top)
                        .round(3))
        rows_for_llm = top_rows.to_dict(orient="records")
        globals_for_llm = {
            "_divert_pct": _divert_pct,
            "_markdown_pct": _markdown_pct,
            "_extra_cost_pct": _extra_cost_pct,
            "intra_share": intra_share,
            "opportunity_pool_usd": opportunity_pool_usd,
        }
        ai_out = _ai_explain(rows_for_llm, globals_for_llm)

        if isinstance(ai_out, dict) and "error" in ai_out:
            st.error(ai_out["error"])
        else:
            with st.expander("AI insight cards", expanded=True):
                try:
                    for row in rows_for_llm:
                        hs2 = row["HS2"]
                        key = hs2  # expect JSON keyed by HS2; fallback to Description
                        item = ai_out.get(key) or ai_out.get(row["Description"]) or {}
                        st.markdown(f"**{row['Description']} (HS{hs2}) — ${row['Realizable_USD_B']:.2f}B**")
                        if item:
                            cols = st.columns(3)
                            cols[0].markdown("- **Reasons**:\n" + "\n".join([f"  - {b}" for b in item.get("reasons", [])]))
                            cols[1].markdown("- **Risks**:\n"   + "\n".join([f"  - {b}" for b in item.get("risks", [])]))
                            cols[2].markdown("- **Actions**:\n" + "\n".join([f"  - {b}" for b in item.get("actions", [])]))
                        else:
                            st.caption("_No AI notes for this row_")
                        st.divider()
                except Exception:
                    st.json(ai_out)  # show raw JSON if formatting fails

    # ---- Optional: HS2 friendly labels (cached, AI) ----
    with st.expander("Optional: Replace descriptions with concise labels (AI)", expanded=False):
        if not view.empty:
            if _ai_available():
                if st.button("Generate concise labels for table/chart"):
                    try:
                        # Create a new 'Pretty' label; do NOT overwrite Description/product_name
                        base_col = "product_name" if "product_name" in view.columns else "Description"
                        view["Pretty"] = view.apply(
                            lambda r: ai_label_hs2(str(r["product_code"]), str(r[base_col])),
                            axis=1
                        )
                        st.success("Labels generated into column: Pretty")
                    except Exception as e:
                        st.error(f"Could not relabel: {e}")
            else:
                st.caption("Set OPENAI_API_KEY to enable AI-based labels.")
        else:
            st.caption("No data loaded yet.")

    # ---- Optional: CFO scenario narrative ----
    with st.expander("Scenario narrative (AI)"):
        # Safe name column for top product
        _name_col = "product_name" if "product_name" in view.columns else ("Description" if "Description" in view.columns else None)
        _top_product = "n/a"
        if _name_col and not view.empty:
            _top_product = str(view.sort_values("Realizable_USD_B", ascending=False)[_name_col].iloc[0])

        metrics = {
          "Divert%": _divert_pct,
          "Markdown%": _markdown_pct,
          "ExtraCost%": _extra_cost_pct,
          "BRICS headroom (B$)": round(opportunity_pool_usd/1e9, 2),
          "Total Realizable (B$)": round(float(view["Realizable_USD_B"].sum()), 3) if "Realizable_USD_B" in view.columns else 0.0,
          "Top product": _top_product,
        }

        if _ai_available():
            if st.button("Generate narrative"):
                text = ai_scenario_narrative(metrics)
                if text:
                    st.info(text)
                else:
                    st.warning("AI narrative not available right now.")
        else:
            st.caption("Set OPENAI_API_KEY to enable the scenario narrative.")

with _tabs[IDX["Cheap Oil Link (Net)"]]:
    import os
    import math
    import pandas as pd
    import streamlit as st
    try:
        import altair as alt
    except Exception:
        alt = None

    st.subheader("Cheap Oil Link — Net Impact to India (Oil + Diversion − Tariff Loss)")

    # --- 1) Inputs pulled from your app / safe defaults ---
    # Monthly saving from Russia discount (USD) → already computed in your app
    try:
        _annual_russia_saving = float(_monthly_russia_saving_usd) * 12.0
    except Exception:
        _annual_russia_saving = 0.0

    # U.S. baseline (X): use your app value, else fallback from env
    try:
        _base_x = float(_annual_us_imports_usd) if _annual_us_imports_usd else float(os.getenv("US_IMPORTS_FROM_INDIA_FALLBACK_USD", 87.3e9))
    except Exception:
        _base_x = float(os.getenv("US_IMPORTS_FROM_INDIA_FALLBACK_USD", 87.3e9))

    # Sliders already defined globally in your app:
    # _divert_pct, _markdown_pct, _extra_cost_pct, _us_tariff_pct, _elasticity

    # --- 2) Safe wrappers around your existing model functions ---
    def _safe_diversion_gain(base_usd: float, divert_pct: float, markdown_pct: float, extra_cost_pct: float) -> float:
        """
        Uses your _diversion_gain() if available; otherwise computes:
        base * divert% * (1 - markdown% - extra_cost%)
        """
        try:
            return float(_diversion_gain(base_usd, divert_pct, markdown_pct, extra_cost_pct))
        except Exception:
            factor = max(0.0, 1.0 - (float(markdown_pct) + float(extra_cost_pct)) / 100.0)
            return float(base_usd) * (float(divert_pct) / 100.0) * factor

    def _safe_tariff_loss(base_usd: float, us_tariff_pct: float, elasticity: float) -> float:
        """
        Uses your _tariff_loss() if available; otherwise uses isoelastic approx:
        loss = base * (1 - (1 + τ)^(-ε)), where τ = tariff%, ε = elasticity (default 1.5 if missing)
        """
        try:
            return float(_tariff_loss(base_usd, us_tariff_pct, elasticity))
        except Exception:
            tau = max(0.0, float(us_tariff_pct) / 100.0)
            eps = float(elasticity) if elasticity not in (None, "") else 1.5
            try:
                return float(base_usd) * (1.0 - (1.0 + tau) ** (-eps))
            except Exception:
                # very conservative fallback
                return float(base_usd) * tau * 0.5

    # --- 3) Components ---
    _div_gain = _safe_diversion_gain(_base_x, _divert_pct, _markdown_pct, _extra_cost_pct)
    _loss     = _safe_tariff_loss(_base_x, _us_tariff_pct, _elasticity)

    # New net: Oil saving + Diversion gain − Tariff loss
    _net = _annual_russia_saving + _div_gain - _loss

    # --- 4) Metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Annual saving from Russia discount", f"${_annual_russia_saving/1e9:,.2f} B")
    m2.metric("Diversion gain (after markdown/cost)", f"+${_div_gain/1e9:,.2f} B")
    m3.metric("Annual U.S. tariff loss (model)", f"-${_loss/1e9:,.2f} B")
    m4.metric("Net impact (oil + diversion − loss)", f"${_net/1e9:,.2f} B")

    st.caption(
        "Formula: Net = (Russia barrels × 12 × discount) + (U.S.-bound × divert% × (1 − markdown − extra cost)) − TariffLoss(X, τ, ε)."
    )

    # --- 5) Optional breakdown table & chart ---
    try:
        _rows = [
            {"Component": "Oil saving (annualized)", "USD_B": _annual_russia_saving / 1e9},
            {"Component": "Diversion gain",          "USD_B": _div_gain / 1e9},
            {"Component": "Tariff loss",             "USD_B": -_loss / 1e9},
            {"Component": "Net impact",              "USD_B": _net / 1e9},
        ]
        _df = pd.DataFrame(_rows)

        st.dataframe(_df, use_container_width=True, height=200)

        if alt:
            st.altair_chart(
                alt.Chart(_df).mark_bar().encode(
                    x=alt.X("Component:N", sort=["Oil saving (annualized)", "Diversion gain", "Tariff loss", "Net impact"]),
                    y=alt.Y("USD_B:Q", title="USD (Billions)"),
                    tooltip=["Component", "USD_B"]
                ).properties(height=280),
                use_container_width=True
            )
    except Exception:
        pass

    # --- 6) Sensitivity (optional quick what-if) ---
    with st.expander("Quick sensitivity (local only)"):
        s1, s2 = st.columns(2)
        _tau_test = s1.slider("Tariff rate τ (%)", 0.0, 25.0, float(_us_tariff_pct), 0.5)
        _eps_test = s2.slider("Elasticity ε", 0.5, 3.0, float(_elasticity) if _elasticity not in (None, "") else 1.5, 0.1)
        _loss_test = _safe_tariff_loss(_base_x, _tau_test, _eps_test)
        st.write(f"Tariff loss at τ={_tau_test:.1f}% and ε={_eps_test:.1f}: **${_loss_test/1e9:,.2f} B**")


with _tabs[IDX["Diversion Calculator"]]:
    st.subheader("Diversion Calculator — move U.S.-bound to BRICS")
    if _annual_us_imports_usd:
        _base_x = float(_annual_us_imports_usd)
    else:
        _base_x = float(os.getenv("US_IMPORTS_FROM_INDIA_FALLBACK_USD", 87.3e9))
    _gain = _diversion_gain(_base_x, _divert_pct, _markdown_pct, _extra_cost_pct)
    _loss = _tariff_loss(_base_x, _us_tariff_pct, _elasticity)
    _net_diversion = _gain - _loss
    c1, c2, c3 = st.columns(3)
    c1.metric("U.S.-bound (annualized)", f"${_base_x/1e9:,.2f} B")
    c2.metric("Recaptured via BRICS", f"${_gain/1e9:,.2f} B")
    c3.metric("Net (diversion − loss)", f"${_net_diversion/1e9:,.2f} B")
    if alt and pd:
        df = pd.DataFrame([
            {"Item": "U.S.-bound", "USD_B": _base_x/1e9},
            {"Item": "Diverted to BRICS", "USD_B": _gain/1e9},
            {"Item": "Tariff Loss", "USD_B": -_loss/1e9},
            {"Item": "Net", "USD_B": _net_diversion/1e9},
        ])
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Item:N"),
            y=alt.Y("USD_B:Q", title="USD (B)"),
            tooltip=["Item", "USD_B"],
        )
        st.altair_chart(chart, use_container_width=True)
    st.caption("Diverted USD = U.S.-bound × divert% × (1 − markdown − extra cost).")

with _tabs[IDX["Guardrails & Compliance"]]:
    st.subheader("Guardrails & Compliance (read this out)")
    st.markdown(
        """
        - **Sanctions & banking**: Russia/Iran trades must comply with all sanctions and payment rules. Expect banking delays.
        - **Rules of origin**: Re-exports via hubs (e.g., UAE) must meet transformation rules — no simple relabeling.
        - **Market access**: Pharma/food need registrations and SPS approvals; plan timelines.
        - **Transparency**: Keep Russia share visibly **below a set cap** to support the case for **≤25% U.S. tariffs**.
        """
    )

with _tabs[IDX["BRICS Diversion Planner"]]:
    import os, json, time
    import pandas as pd
    import streamlit as st

    st.subheader("BRICS Diversion Planner — GenAI-guided settings")

    # ------------------ Context pulled from app / defaults ------------------
    # Existing BRICS totals (B$) used in other tabs; update if your app uses a different store
    _brics_defaults = {
        "China": 2580.0, "UAE": 539.0, "Brazil": 278.0, "Saudi Arabia": 250.0,
        "South Africa": 101.0, "Egypt": 95.0, "Russia": 270.0, "Iran": 35.0, "Ethiopia": 23.0,
    }
    brics_markets = st.session_state.get("brics_markets", _brics_defaults)
    brics_list = list(brics_markets.keys())

    # ------------------ Inputs: risk signals & targeting ------------------
    cA, cB = st.columns([1, 1])
    with cA:
        target_countries = st.multiselect(
            "Target BRICS markets", brics_list,
            default=["UAE", "Saudi Arabia", "Brazil"]
        )
        baseline_us_tariff_pct = st.slider("Baseline U.S. tariff risk (%)", 0.0, 40.0, float(st.session_state.get("_us_tariff_pct", 10.0)), 0.5)
        partner_tariff_pct = st.slider("Avg. partner (BRICS) import tariff on India (%)", 0.0, 40.0, 8.0, 0.5)
        geopolitics_idx = st.slider("Geo-political heat (0 calm → 10 hot)", 0, 10, 6, 1)
    with cB:
        fx_vol_idx = st.slider("FX volatility (0 low → 10 high)", 0, 10, 5, 1)
        logistics_idx = st.slider("Logistics friction (0 smooth → 10 rough)", 0, 10, 4, 1)
        sanctions_idx = st.slider("Sanctions/Compliance risk (0 none → 10 high)", 0, 10, 3, 1)
        notes = st.text_area("Context notes (optional)", placeholder="e.g., RMB liquidity tight; INR SRVAs available; EU demand weak; oil discounts stable…")

    # ------------------ Which sliders this planner will set ------------------
    # These are the canonical keys many of your tabs use; adjust if your slider keys differ.
    # - _divert_pct:    % of U.S.-bound exports you will divert to BRICS
    # - _markdown_pct:  Expected price markdown needed to win BRICS demand
    # - _extra_cost_pct:Extra logistics/compliance costs
    # - opp_intra_share:BRICS intra-supply share (reduces headroom)
    slider_keys = {
        "divert": "_divert_pct",
        "markdown": "_markdown_pct",
        "extra_cost": "_extra_cost_pct",
        "intra_share": "opp_intra_share",
    }

    # ------------------ Helper: apply into session state ------------------
    def apply_reco_to_state(reco: dict):
        # clamp & set; Streamlit will re-run and other tabs can read from session_state
        def _clamp(x, lo=0.0, hi=100.0): 
            try: return max(lo, min(float(x), hi))
            except: return lo
        st.session_state[slider_keys["divert"]]      = _clamp(reco.get("divert_pct", 25))
        st.session_state[slider_keys["markdown"]]    = _clamp(reco.get("markdown_pct", 5))
        st.session_state[slider_keys["extra_cost"]]  = _clamp(reco.get("extra_cost_pct", 3))
        st.session_state[slider_keys["intra_share"]] = _clamp(reco.get("brics_intra_supply_pct", 60))
        # also mirror to globals if your code reads globals (best-effort; might not affect widgets directly)
        globals()["_divert_pct"]     = st.session_state[slider_keys["divert"]]
        globals()["_markdown_pct"]   = st.session_state[slider_keys["markdown"]]
        globals()["_extra_cost_pct"] = st.session_state[slider_keys["extra_cost"]]

    # ------------------ GenAI proposal (uses OPENAI_API_KEY if available) ------------------
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def llm_propose_settings(payload: dict) -> dict:
        """
        Ask an LLM for slider recommendations.
        Returns dict: {divert_pct, markdown_pct, extra_cost_pct, brics_intra_supply_pct, rationale:[...]}
        """
        if not OPENAI_API_KEY:
            return {"error": "OPENAI_API_KEY not set"}

        prompt = f"""
You are a trade strategy assistant. 
Given these inputs, propose percentage settings for a BRICS diversion plan:
- divert_pct (0-100): share of U.S.-bound exports India should divert to BRICS
- markdown_pct (0-100): price markdown needed to win BRICS demand
- extra_cost_pct (0-100): extra logistics/compliance costs vs U.S. route
- brics_intra_supply_pct (0-100): share of BRICS demand already met internally (reduces headroom)
Also return a short 'rationale' list (3 bullets, <=18 words each), and country_notes: dict[country]->1 bullet.

Constraints:
- Keep all percentages in [0, 100].
- Higher tariffs/risks ⇒ lower divert_pct or higher markdown/extra_cost.
- More geopolitics & sanctions risk ⇒ lower divert_pct, higher extra_cost.
- If FX/logistics risk is high, raise extra_cost and slightly lower divert_pct.

Return STRICT JSON with keys:
{{"divert_pct": number, "markdown_pct": number, "extra_cost_pct": number, "brics_intra_supply_pct": number,
  "rationale": ["...","...","..."], "country_notes": {{"UAE":"...", "Brazil":"..."}} }}

INPUTS:
{json.dumps(payload, ensure_ascii=False, indent=2)}
        """.strip()

        try:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                rsp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=0.2,
                    response_format={"type":"json_object"},
                    messages=[
                        {"role": "system", "content": "Return valid JSON only. No prose outside JSON."},
                        {"role": "user", "content": prompt}
                    ],
                )
                content = rsp.choices[0].message.content
            except Exception:
                import openai
                openai.api_key = OPENAI_API_KEY
                rsp = openai.ChatCompletion.create(
                    model=OPENAI_MODEL, temperature=0.2,
                    messages=[
                        {"role":"system","content":"Return valid JSON only. No prose outside JSON."},
                        {"role":"user","content": prompt}
                    ]
                )
                content = rsp["choices"][0]["message"]["content"]

            obj = json.loads(content)
            # sanitize
            for k in ["divert_pct","markdown_pct","extra_cost_pct","brics_intra_supply_pct"]:
                if k in obj:
                    try:
                        obj[k] = float(obj[k])
                    except:
                        obj[k] = None
            if "rationale" not in obj or not isinstance(obj["rationale"], list):
                obj["rationale"] = []
            if "country_notes" not in obj or not isinstance(obj["country_notes"], dict):
                obj["country_notes"] = {}
            return obj
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # ------------------ Heuristic fallback if no LLM / failure ------------------
    def heuristic_proposal(payload: dict) -> dict:
        # Simple interpretable rules as a backstop:
        t_us = float(payload.get("baseline_us_tariff_pct", 10.0))
        t_partner = float(payload.get("partner_tariff_pct", 8.0))
        geo = float(payload.get("geopolitics_idx", 5.0))
        fx  = float(payload.get("fx_vol_idx", 5.0))
        log = float(payload.get("logistics_idx", 5.0))
        sanc = float(payload.get("sanctions_idx", 3.0))
        # Start from midpoints:
        divert = 25.0 + max(0.0, 12.0 - 0.6*t_us - 0.4*t_partner) - (0.8*geo + 0.8*sanc + 0.6*fx + 0.6*log)
        markdown = 5.0 + 0.5*t_partner + 0.4*geo + 0.4*fx
        extra = 3.0 + 0.6*log + 0.4*sanc + 0.3*fx
        intra = 60.0 + 0.2*geo + 0.3*sanc - 0.1*len(payload.get("target_countries", []))
        # clamp
        clamp = lambda x: max(0.0, min(float(x), 100.0))
        return {
            "divert_pct": clamp(divert),
            "markdown_pct": clamp(markdown),
            "extra_cost_pct": clamp(extra),
            "brics_intra_supply_pct": clamp(intra),
            "rationale": [
                "Adjusted for tariffs and partner frictions.",
                "Higher geopolitics/sanctions reduce diversion and lift costs.",
                "FX/logistics raise extra cost; markdown rises with partner tariffs.",
            ],
            "country_notes": {c: "Watch tariff corridors and FX clearing." for c in payload.get("target_countries", [])}
        }

    # ------------------ Build request payload ------------------
    payload = {
        "target_countries": target_countries,
        "baseline_us_tariff_pct": baseline_us_tariff_pct,
        "partner_tariff_pct": partner_tariff_pct,
        "geopolitics_idx": geopolitics_idx,
        "fx_vol_idx": fx_vol_idx,
        "logistics_idx": logistics_idx,
        "sanctions_idx": sanctions_idx,
        "brics_market_sizes_B": {k: float(v) for k, v in brics_markets.items()},
        "notes": notes or "",
    }

    # ------------------ Run GenAI / fallback ------------------
    cL, cR = st.columns([1, 1])
    with cL:
        go_llm = st.button("🔮 Generate plan with GenAI", use_container_width=True, disabled=(not bool(OPENAI_API_KEY)))
        if not OPENAI_API_KEY:
            st.caption("⚠️ Set OPENAI_API_KEY in your .env to enable GenAI planning.")
    with cR:
        go_heur = st.button("🧮 Generate plan (heuristic)", use_container_width=True)

    reco = {}
    if go_llm:
        with st.spinner("Asking GenAI for recommendations…"):
            reco = llm_propose_settings(payload)
            if "error" in reco:
                st.error(reco["error"])
                reco = {}
    if go_heur:
        reco = heuristic_proposal(payload)

    # ------------------ Show / Apply recommendations ------------------
    def _show_reco(reco: dict):
        if not reco: 
            return
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Divert %", f"{reco.get('divert_pct', 0):.1f}%")
        c2.metric("Markdown %", f"{reco.get('markdown_pct', 0):.1f}%")
        c3.metric("Extra cost %", f"{reco.get('extra_cost_pct', 0):.1f}%")
        c4.metric("BRICS intra-supply %", f"{reco.get('brics_intra_supply_pct', 0):.1f}%")

        if reco.get("rationale"):
            st.markdown("**Why these settings**")
            st.markdown("\n".join([f"- {b}" for b in reco["rationale"]]))
        if reco.get("country_notes"):
            st.markdown("**Country notes**")
            for k, v in reco["country_notes"].items():
                st.markdown(f"- **{k}**: {v}")

        if st.button("✅ Apply to sliders", type="primary"):
            apply_reco_to_state(reco)
            st.success("Applied. Re-run dependent tabs (e.g., Opportunities) will use these values.")

    _show_reco(reco)

    # ------------------ Advanced: map to your custom slider keys (optional) ------------------
    with st.expander("Advanced: map to custom slider keys"):
        k_div = st.text_input("Key for divert %", value=slider_keys["divert"])
        k_mark = st.text_input("Key for markdown %", value=slider_keys["markdown"])
        k_cost = st.text_input("Key for extra cost %", value=slider_keys["extra_cost"])
        k_intra = st.text_input("Key for BRICS intra-supply %", value=slider_keys["intra_share"])
        if st.button("Save key mapping"):
            slider_keys["divert"] = k_div.strip() or slider_keys["divert"]
            slider_keys["markdown"] = k_mark.strip() or slider_keys["markdown"]
            slider_keys["extra_cost"] = k_cost.strip() or slider_keys["extra_cost"]
            slider_keys["intra_share"] = k_intra.strip() or slider_keys["intra_share"]
            st.success("Key mapping updated. Future 'Apply' uses these keys.")


# EOF

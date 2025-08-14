"""
backend_oil_import_optimizer.py — UPDATED

What’s new in this version
- Supports an explicit Russian Oil Discount % (percent of Brent) passed from the Frontend.
- Implements your requested rule:
  If the UI setting **Minimum non-Russia share** is X (e.g., 0.50), then we **force Russia share = 1 - X**
  (e.g., 50%), and allocate the **remaining X** across other suppliers by effective cost.
- Keeps the original structure the frontend expects: run_full_pipeline(...) returns quotes, mix, metrics, etc.

This is a self-contained backend. Drop it next to the frontend file.
"""
from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Optional runtime deps; code works without them using fallbacks
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import feedparser  # for simple headline sentiment hook (kept minimal)
except Exception:
    feedparser = None

# OpenAI (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------------
# Constants & simple data tables
# -----------------------------
SUPPLIERS: List[str] = ["UAE", "US", "Saudi Arabia", "Iraq", "Russia"]

# Which benchmark each supplier maps to
SUPPLIER_BENCHMARK: Dict[str, str] = {
    "UAE": "Dubai",
    "Saudi Arabia": "Dubai",
    "US": "Brent",
    "Iraq": "Brent",
    "Russia": "Urals",
}

# Static price adjustments per supplier (USD/bbl) — tweak as needed
SUPPLIER_PRICE_ADJ: Dict[str, float] = {
    "UAE": +1.0,
    "Saudi Arabia": +1.5,
    "US": +2.0,
    "Iraq": +0.5,
    "Russia": 0.0,  # base before the discount slider applies
}

# Carbon intensity (kgCO2e/bbl) — demo values
CARBON_INTENSITY: Dict[str, float] = {
    "UAE": 36.0,
    "Saudi Arabia": 35.0,
    "US": 44.0,
    "Iraq": 40.0,
    "Russia": 39.0,
}

# ESG-ish score (0..100) — demo values
ESG_SCORE: Dict[str, float] = {
    "UAE": 62,
    "Saudi Arabia": 60,
    "US": 70,
    "Iraq": 45,
    "Russia": 40,
}

# Freight/handling/insurance demo params
DEFAULT_FREIGHT_BASE = 1.50              # USD/bbl
DEFAULT_FREIGHT_PER_1000NM = 0.60        # USD/bbl per 1000 nautical miles
DEFAULT_HANDLING = 0.40                  # USD/bbl
DEFAULT_INSURANCE_RATE = 0.0025          # as a fraction of raw price (0.25%)

# Fallback benchmark prices (USD/bbl) if live fails
BASE_BENCHMARK_FALLBACK = {"Brent": 85.0, "Dubai": 83.0, "Urals": 80.0}

# Export ports (lat, lon) — rough demo coords
SUPPLIER_EXPORT_PORT: Dict[str, Tuple[float, float, str]] = {
    "UAE": (25.2697, 55.3095, "Fujairah"),            # UAE oil hub on Gulf of Oman
    "Saudi Arabia": (26.6400, 50.1590, "Ras Tanura"),  # KSA east coast
    "US": (27.8006, -97.3964, "Corpus Christi"),       # US Gulf
    "Iraq": (30.4966, 47.8190, "Basra"),               # Al Basra Oil Terminal
    "Russia": (44.7244, 37.7689, "Novorossiysk"),      # Black Sea
}

# Indian destination ports (lat, lon)
INDIA_PORTS: Dict[str, Tuple[float, float]] = {
    "India-Mundra": (22.7359, 69.7036),
    "India-JNPT": (18.9479, 72.9337),
    "India-Paradip": (20.3167, 86.6167),
}

GRID_STEP = 0.05  # 5% increments for mixes

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Scenario:
    tariff_rate: float
    demand_bpd: int
    max_supplier_share: float
    min_non_russia_share: float
    selected_indian_port: str

@dataclass
class SupplierQuote:
    supplier: str
    benchmark: str
    raw_price: float
    tariff_component: float
    freight: float
    handling: float
    insurance: float
    landed_cost: float
    carbon_intensity: float
    esg: float

# -----------------------------
# Helpers
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def round_grid(x: float, step: float = GRID_STEP) -> float:
    return round(x / step) * step


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in nautical miles."""
    R_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    km = R_km * c
    nm = km * 0.539957
    return nm

# -----------------------------
# Live / fallback data
# -----------------------------

def fetch_live_benchmarks() -> Dict[str, float]:
    brent = BASE_BENCHMARK_FALLBACK["Brent"]
    dubai = BASE_BENCHMARK_FALLBACK["Dubai"]
    urals = BASE_BENCHMARK_FALLBACK["Urals"]

    # Brent via yfinance if available
    try:
        if yf is not None:
            t = yf.Ticker("BZ=F")
            px = t.history(period="1d").get("Close")
            if px is not None and len(px) > 0:
                brent = float(px.iloc[-1])
    except Exception:
        pass

    # (Dubai/Urals live sources omitted for simplicity here; keep fallbacks)
    return {"Brent": brent, "Dubai": dubai, "Urals": urals}

# -----------------------------
# Pricing components
# -----------------------------

def price_for_supplier(supplier: str, live_bench: Dict[str, float]) -> float:
    bmk = SUPPLIER_BENCHMARK[supplier]
    base = live_bench.get(bmk, BASE_BENCHMARK_FALLBACK[bmk])
    adj = SUPPLIER_PRICE_ADJ[supplier]
    return base + adj


def tariff_component(raw_price: float, tariff_rate: float) -> float:
    return raw_price * clamp(tariff_rate, 0.0, 1.0)


def supplier_port(supplier: str) -> str:
    return SUPPLIER_EXPORT_PORT.get(supplier, (0.0, 0.0, "Unknown"))[2]


def freight_for_path(supplier: str, indian_port_key: str) -> float:
    s_lat, s_lon, _ = SUPPLIER_EXPORT_PORT[supplier]
    d_lat, d_lon = INDIA_PORTS[indian_port_key]
    nm = haversine_nm(s_lat, s_lon, d_lat, d_lon)
    return DEFAULT_FREIGHT_BASE + DEFAULT_FREIGHT_PER_1000NM * (nm / 1000.0)


def insurance_component(raw_price: float) -> float:
    return raw_price * DEFAULT_INSURANCE_RATE


def landed_cost(raw: float, t: float, f: float, h: float, ins: float) -> float:
    return raw + t + f + h + ins

# -----------------------------
# Quotes & sentiment
# -----------------------------

def risk_score_from_sentiment(base_esg: float, sentiment: float) -> float:
    return clamp(base_esg + 10.0 * sentiment, 0.0, 100.0)


def quote_supplier(
    supplier: str,
    scenario: Scenario,
    live_bench: Dict[str, float],
    russia_discount_pct: float = 0.0,
) -> SupplierQuote:
    raw_price = price_for_supplier(supplier, live_bench)

    # Apply Russian discount as % of Brent if applicable
    if supplier.lower().startswith("russia") and russia_discount_pct > 0.0:
        brent = live_bench.get("Brent", BASE_BENCHMARK_FALLBACK["Brent"])
        raw_price = max(0.0, raw_price - brent * (russia_discount_pct / 100.0))

    t = tariff_component(raw_price, scenario.tariff_rate)
    f = freight_for_path(supplier, scenario.selected_indian_port)
    h = DEFAULT_HANDLING
    ins = insurance_component(raw_price)
    lc = landed_cost(raw_price, t, f, h, ins)

    return SupplierQuote(
        supplier=supplier,
        benchmark=SUPPLIER_BENCHMARK[supplier],
        raw_price=raw_price,
        tariff_component=t,
        freight=f,
        handling=h,
        insurance=ins,
        landed_cost=lc,
        carbon_intensity=CARBON_INTENSITY[supplier],
        esg=ESG_SCORE[supplier],
    )


def scenario_quotes(
    scenario: Scenario,
    live_bench: Dict[str, float],
    russia_discount_pct: float = 0.0,
) -> Dict[str, SupplierQuote]:
    return {
        s: quote_supplier(s, scenario, live_bench, russia_discount_pct=russia_discount_pct)
        for s in SUPPLIERS
    }

# -----------------------------
# Optimizer (with forced Russia share rule)
# -----------------------------

def _effective_cost(q: SupplierQuote, objective: str, sentiment: float = 0.0) -> float:
    """Convert landed cost + risk/carbon into a single comparable number."""
    # Blend weights (tunable)
    if objective == "cost":
        wc, wcarb, wesg = 1.0, 0.0, 0.0
    elif objective == "risk":
        wc, wcarb, wesg = 0.7, 0.1, -0.2   # higher ESG reduces effective cost
    else:  # hybrid
        wc, wcarb, wesg = 0.8, 0.1, -0.1

    # Adjust ESG with sentiment
    esg_adj = risk_score_from_sentiment(q.esg, sentiment)
    return (
        wc * q.landed_cost
        + wcarb * (q.carbon_intensity / 10.0)
        + wesg * (esg_adj / 10.0)
    )


def optimize_mix(
    scenario: Scenario,
    quotes: Dict[str, SupplierQuote],
    objective: str = "hybrid",
    sentiment: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Enforce the rule you asked for, exactly:
      - **Russia share = 1 − min_non_russia_share** (no cap and no rounding on Russia).
      - The **remainder** goes to non‑Russia suppliers by AI (effective cost) and respects
        `max_supplier_share` *if feasible*.
      - If caps make the remainder infeasible to allocate, we **relax caps** on the
        cheapest non‑Russia supplier(s) to absorb the leftover, keeping Russia fixed.
    """
    # 1) Force Russia share exactly as the complement of the non‑Russia minimum
    forced_russia_share = clamp(1.0 - scenario.min_non_russia_share, 0.0, 1.0)

    # 2) Remaining share to allocate among non‑Russia
    remaining = clamp(1.0 - forced_russia_share, 0.0, 1.0)

    non_russia = [s for s in SUPPLIERS if not s.lower().startswith("russia")]

    # Rank by effective cost (cheapest first)
    ranked = sorted(
        non_russia,
        key=lambda s: _effective_cost(quotes[s], objective, sentiment),
    )

    # Initialize mix with Russia fixed
    mix: Dict[str, float] = {s: 0.0 for s in SUPPLIERS}
    mix["Russia"] = forced_russia_share

    # 3) Greedy fill within caps (continuous shares; no 5% rounding)
    cap = clamp(scenario.max_supplier_share, 0.0, 1.0)
    remain = remaining
    for s in ranked:
        if remain <= 1e-12:
            break
        room = max(0.0, cap - mix[s])
        take = min(room, remain)
        mix[s] += take
        remain -= take

    # 4) If still leftover because caps were too tight, relax caps on cheapest suppliers
    if remain > 1e-12:
        for s in ranked:
            if remain <= 1e-12:
                break
            mix[s] += remain
            remain = 0.0

    # At this point, Russia is fixed and others sum to the remainder; total should be 1.0
    # Compute weighted metrics
    w_cost = sum(mix[s] * quotes[s].landed_cost for s in SUPPLIERS)
    w_carb = sum(mix[s] * quotes[s].carbon_intensity for s in SUPPLIERS)
    w_esg = sum(mix[s] * quotes[s].esg for s in SUPPLIERS)

    metrics = {
        "weighted_landed_cost_usd_bbl": w_cost,
        "weighted_carbon_kgco2e_bbl": w_carb,
        "weighted_esg": w_esg,
    }
    return mix, metrics

# -----------------------------
# Sentiment & advisory (minimal)
# -----------------------------

def sentiment_from_feeds() -> Tuple[float, List[str]]:
    if feedparser is None:
        return 0.0, []
    urls = [
        "https://news.google.com/rss/search?q=india+oil+imports",
        "https://news.google.com/rss/search?q=shipping+tankers",
    ]
    headlines: List[str] = []
    for u in urls:
        try:
            f = feedparser.parse(u)
            for e in f.entries[:5]:
                headlines.append(e.title)
        except Exception:
            pass
    # toy sentiment: neutral unless many headlines
    sent = 0.0
    if any("tension" in h.lower() or "sanction" in h.lower() for h in headlines):
        sent = -0.2
    return sent, headlines


def genai_advisory(
    scenario: Scenario,
    mix: Dict[str, float],
    quotes: Dict[str, SupplierQuote],
    metrics: Dict[str, float],
    sentiment: float,
) -> Tuple[str, bool]:
    client = None
    try:
        if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            client = OpenAI()
    except Exception:
        client = None

    if client is None:
        # Offline template
        lines = [
            "Recommended Supplier Blend:",
            *[f"- {s}: {mix.get(s,0.0)*100:.1f}%" for s in SUPPLIERS],
            "Rationale:",
            f"- Cost: Weighted landed cost ≈ ${metrics['weighted_landed_cost_usd_bbl']:.2f}/bbl.",
            f"- Risk: Headline sentiment {sentiment:+.2f}; diversified exposure.",
            f"- Carbon: ~{metrics['weighted_carbon_kgco2e_bbl']:.1f} kgCO2e/bbl.",
        ]
        return "\n".join(lines), False

    # Simple live call
    try:
        prompt = (
            "You are an oil trade advisor. Summarize the supplier mix and why it makes sense "
            "for India in 5-7 bullets, using cost, risk, carbon."
        )
        content = [
            {"role": "system", "content": "Be concise and factual."},
            {"role": "user", "content": prompt + f"\nMix: {mix}\nMetrics: {metrics}\nSentiment: {sentiment}"},
        ]
        resp = client.chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=content)
        txt = resp.choices[0].message.content.strip()
        return txt, True
    except Exception:
        return "(GenAI advisory unavailable; showing offline template.)", False

# -----------------------------
# Public API used by the frontend
# -----------------------------

def shipping_time_days(supplier: str, indian_port_key: str, speed_knots: float) -> Tuple[float, float]:
    s_lat, s_lon, _ = SUPPLIER_EXPORT_PORT[supplier]
    d_lat, d_lon = INDIA_PORTS[indian_port_key]
    nm = haversine_nm(s_lat, s_lon, d_lat, d_lon)
    days = nm / (max(8.0, float(speed_knots)) * 24.0)
    return nm, days


def run_full_pipeline(
    tariff_rate: float,
    demand_bpd: int,
    max_supplier_share: float,
    min_non_russia_share: float,
    selected_indian_port: str,
    optimizer_objective: str = "hybrid",
    russia_discount_pct: float = 0.0,
) -> Dict[str, object]:
    # Assemble scenario
    sc = Scenario(
        tariff_rate=float(tariff_rate),
        demand_bpd=int(demand_bpd),
        max_supplier_share=float(max_supplier_share),
        min_non_russia_share=float(min_non_russia_share),
        selected_indian_port=selected_indian_port,
    )

    # Live/fallback benchmarks
    live_bench = fetch_live_benchmarks()

    # Quotes per supplier
    quotes = scenario_quotes(sc, live_bench, russia_discount_pct=russia_discount_pct)

    # Optimize mix with forced Russia share rule
    news_sentiment, headlines = sentiment_from_feeds()
    mix, metrics = optimize_mix(sc, quotes, objective=optimizer_objective, sentiment=news_sentiment)

    # Advisory
    advisory, used_llm = genai_advisory(sc, mix, quotes, metrics, news_sentiment)

    return {
        "scenario": sc,
        "live_benchmarks": live_bench,
        "quotes": quotes,
        "mix": mix,
        "metrics": metrics,
        "news_sentiment": news_sentiment,
        "headlines": headlines,
        "advisory": advisory,
        "advisory_used_llm": used_llm,
    }

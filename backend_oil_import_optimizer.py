# backend_import_strategy.py
# Real-time backend for AI-Powered Oil Import Optimizer (India)
# - Brent (intraday) via yfinance (BZ=F)
# - Dubai via IMF PCPS (monthly) using DB.NOMICS (no key)
# - Urals via TradingEconomics (optional key), else Brent - spread fallback
# - RSS headline sentiment via feedparser
# - OpenAI advisory if OPENAI_API_KEY is set (fallback to offline template)

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math, random, os, json
from datetime import datetime, timezone

# Optional dependencies (each handled with safe fallbacks)
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import feedparser
except Exception:
    feedparser = None

try:
    import requests
except Exception:
    requests = None

# OpenAI (optional)
OPENAI_ENABLED = False
_client = None
try:
    from openai import OpenAI
    if os.getenv("OPENAI_API_KEY"):
        _client = OpenAI()
        OPENAI_ENABLED = True
except Exception:
    OPENAI_ENABLED = False
    _client = None

# ------------------------------
# Static reference / defaults
# ------------------------------

SUPPLIERS = ["Russia", "Iraq", "Saudi Arabia", "UAE", "US", "Nigeria"]

SUPPLIER_BENCHMARK = {
    "Russia": "Urals",
    "Iraq": "Dubai",
    "Saudi Arabia": "Brent",
    "UAE": "Dubai",
    "US": "Brent",
    "Nigeria": "Brent",
}

# Fallback base prices if all live routes fail
BASE_BENCHMARK_FALLBACK = {"Brent": 82.0, "Dubai": 79.0, "Urals": 74.0}

# Supplier differentials (USD/bbl) on top of benchmark
SUPPLIER_PRICE_ADJ = {
    "Russia": -3.0, "Iraq": 0.5, "Saudi Arabia": 1.5,
    "UAE": 1.0, "US": 2.0, "Nigeria": 1.0,
}

# Freight/handling/insurance (allow env overrides)
DEFAULT_FREIGHT_BASE = float(os.getenv("FREIGHT_BASE", "1.8"))
DEFAULT_FREIGHT_PER_1000NM = float(os.getenv("FREIGHT_PER_1000NM", "0.9"))
DEFAULT_HANDLING = float(os.getenv("HANDLING_USD", "0.6"))
DEFAULT_INSURANCE_RATE = float(os.getenv("INSURANCE_RATE", "0.0025"))  # 0.25%

# Carbon & ESG (static for demo)
CARBON_INTENSITY = {"Russia":37.0, "Iraq":42.0, "Saudi Arabia":34.0, "UAE":36.0, "US":45.0, "Nigeria":40.0}
ESG_SCORE       = {"Russia":42,   "Iraq":40,   "Saudi Arabia":55,    "UAE":67,   "US":75,  "Nigeria":50}

# Very approximate port positions (demo)
PORTS = {
    "India-Mundra": (22.74, 69.71), "India-JNPT": (18.95, 72.95), "India-Paradip": (20.27, 86.68),
    "Russia-Novorossiysk": (44.72, 37.77), "Iraq-Basra": (30.51, 47.82), "Saudi-RasTanura": (26.64, 50.16),
    "UAE-Ruwais": (24.12, 52.73), "US-Houston": (29.73, -95.27), "Nigeria-Bonny": (4.45, 7.17),
}

NEWS_FEEDS_DEFAULT = [
    "https://news.google.com/rss/search?q=oil+prices",
    "https://news.google.com/rss/search?q=shipping+freight+tankers",
    "https://news.google.com/rss/search?q=india+oil+imports",
]

NEG_WORDS = ["sanction","strike","conflict","war","attack","embargo","tariff","ban","dispute","seizure","escalation"]
POS_WORDS = ["deal","agreement","stable","resume","increase","safe","secure","de-escalation","reopen","truce"]

# ------------------------------
# Data Types
# ------------------------------

@dataclass
class Scenario:
    tariff_rate: float
    demand_bpd: int
    max_supplier_share: float
    min_non_russia_share: float
    selected_indian_port: str
    fx_rate: float = 83.0  # INR/USD (not used in USD outputs)

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
    esg: int

# ------------------------------
# Time & Math Utilities
# ------------------------------

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def haversine_nm(lat1, lon1, lat2, lon2):
    R_km = 6371.0; km_to_nm = 0.539957
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    dist_km = R_km * c
    return dist_km * km_to_nm

def supplier_port(supplier: str) -> str:
    return {
        "Russia": "Russia-Novorossiysk", "Iraq": "Iraq-Basra", "Saudi Arabia": "Saudi-RasTanura",
        "UAE": "UAE-Ruwais", "US": "US-Houston", "Nigeria": "Nigeria-Bonny",
    }[supplier]

# ------------------------------
# Live Data Fetchers
# ------------------------------

def fetch_dbnomics_imf_pcps_last_price(code: str) -> Optional[float]:
    """
    Fetch last value for IMF Primary Commodity Prices via DB.NOMICS (no key).
    Codes: POILBRE (Brent), POILDUB (Dubai), POILWTI (WTI)
    """
    if requests is None:
        return None
    url = f"https://api.db.nomics.world/v22/series/IMF/PCPS/{code}?observations=1"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        series = data["series"]["docs"][0]
        val = series["values"][-1]
        return float(val)
    except Exception:
        return None

def fetch_tradingeconomics_urals() -> Optional[float]:
    """
    Optional: real-time Urals via TradingEconomics API.
    Requires TRADING_ECONOMICS_KEY in .env (format: email:key).
    """
    if requests is None:
        return None
    key = os.getenv("TRADING_ECONOMICS_KEY")
    if not key:
        return None
    try:
        url = f"https://api.tradingeconomics.com/commodities/urals?c={key}&format=json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        arr = r.json()
        for item in arr:
            if "Last" in item and item["Last"] is not None:
                return float(item["Last"])
        return None
    except Exception:
        return None

def fetch_live_benchmarks() -> Dict[str, float]:
    """
    Returns {'Brent': ..., 'Dubai': ..., 'Urals': ...}
    Priority:
      Brent: yfinance BZ=F (intraday). Fallback -> IMF PCPS (POILBRE), then local fallback
      Dubai: IMF PCPS (POILDUB). Fallback -> Brent - DUBAI_SPREAD_USD (default 3)
      Urals: TradingEconomics (if key). Fallback -> Brent - URALS_SPREAD_USD (default 6)
    ENV overrides: PRICE_BRENT / PRICE_DUBAI / PRICE_URALS (if provided, force values)
    """
    # ENV overrides (if set)
    out: Dict[str, float] = {}
    for k in ["Brent","Dubai","Urals"]:
        v = os.getenv(f"PRICE_{k.upper()}")
        if v:
            try: out[k] = float(v)
            except: pass

    # Brent
    brent_val = out.get("Brent")
    if brent_val is None:
        # yfinance intraday
        if yf is not None:
            try:
                v = yf.Ticker("BZ=F").fast_info.last_price
                if v: brent_val = float(v)
            except Exception:
                brent_val = None
        # IMF monthly (fallback)
        if brent_val is None:
            brent_val = fetch_dbnomics_imf_pcps_last_price("POILBRE")
    if brent_val is None:
        brent_val = BASE_BENCHMARK_FALLBACK["Brent"]
    out["Brent"] = brent_val

    # Dubai (IMF)
    dubai_val = out.get("Dubai")
    if dubai_val is None:
        dubai_val = fetch_dbnomics_imf_pcps_last_price("POILDUB")
    if dubai_val is None:
        spread = float(os.getenv("DUBAI_SPREAD_USD", "3.0"))
        dubai_val = brent_val - spread
    out["Dubai"] = max(30.0, dubai_val)

    # Urals (TradingEconomics)
    urals_val = out.get("Urals")
    if urals_val is None:
        te_val = fetch_tradingeconomics_urals()
        if te_val is not None:
            urals_val = te_val
        else:
            spread = float(os.getenv("URALS_SPREAD_USD", "6.0"))
            urals_val = brent_val - spread
    out["Urals"] = max(25.0, urals_val)

    return out

# ------------------------------
# Sentiment / Risk
# ------------------------------

def simple_sentiment_score(text: str) -> float:
    t = (text or "").lower()
    score = 0
    for w in NEG_WORDS:
        if w in t: score -= 1
    for w in POS_WORDS:
        if w in t: score += 1
    return max(-1.0, min(1.0, score/6.0))

def fetch_news_headlines(max_items: int = 15) -> List[str]:
    feeds_env = os.getenv("NEWS_FEEDS")
    if feeds_env:
        sources = [u.strip() for u in feeds_env.split(",") if u.strip()]
    else:
        sources = NEWS_FEEDS_DEFAULT

    if feedparser is None:
        return []

    headlines = []
    try:
        for url in sources:
            d = feedparser.parse(url)
            for e in d.entries[: max(5, max_items//len(sources) or 5)]:
                title = getattr(e, "title", "")
                if title:
                    headlines.append(title)
        return headlines[:max_items]
    except Exception:
        return []

def sentiment_from_feeds() -> Tuple[float, List[str]]:
    titles = fetch_news_headlines()
    if not titles:
        return 0.0, []
    total = sum(simple_sentiment_score(t) for t in titles)
    avg = total / max(1, len(titles))
    return avg, titles

def risk_score_from_sentiment(base_esg: int, sentiment: float) -> float:
    return max(0.0, min(100.0, base_esg + 10.0*sentiment))

# ------------------------------
# Pricing & Logistics
# ------------------------------

def price_for_supplier(supplier: str, live_bench: Dict[str,float]) -> float:
    bench = SUPPLIER_BENCHMARK[supplier]
    base = live_bench.get(bench, BASE_BENCHMARK_FALLBACK[bench])
    adj = SUPPLIER_PRICE_ADJ[supplier]
    jitter = random.uniform(-0.6, 0.6)  # tiny day swing
    return base + adj + jitter

def insurance_component(price_usd_bbl: float) -> float:
    return price_usd_bbl * DEFAULT_INSURANCE_RATE

def tariff_component(price_usd_bbl: float, tariff_rate: float) -> float:
    return price_usd_bbl * tariff_rate

def landed_cost(price_usd_bbl: float, tariff_usd: float, freight: float, handling: float, insurance: float) -> float:
    return price_usd_bbl + tariff_usd + freight + handling + insurance

def freight_for_path(supplier: str, indian_port: str) -> float:
    sp = supplier_port(supplier)
    lat1, lon1 = PORTS[sp]; lat2, lon2 = PORTS[indian_port]
    d_nm = haversine_nm(lat1, lon1, lat2, lon2)
    return DEFAULT_FREIGHT_BASE + DEFAULT_FREIGHT_PER_1000NM * (d_nm/1000.0)

def shipping_time_days(supplier: str, indian_port: str, speed_knots: float = 14.0) -> Tuple[float, float]:
    sp = supplier_port(supplier)
    lat1, lon1 = PORTS[sp]; lat2, lon2 = PORTS[indian_port]
    d_nm = haversine_nm(lat1, lon1, lat2, lon2)
    hours = d_nm / max(1e-6, speed_knots)
    return d_nm, hours/24.0

# ------------------------------
# Core Model
# ------------------------------

def quote_supplier(supplier: str, scenario: Scenario, live_bench: Dict[str,float]) -> SupplierQuote:
    raw_price = price_for_supplier(supplier, live_bench)
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

def scenario_quotes(scenario: Scenario, live_bench: Dict[str,float]) -> Dict[str, SupplierQuote]:
    return {s: quote_supplier(s, scenario, live_bench) for s in SUPPLIERS}

def optimize_mix(
    scenario: Scenario,
    quotes: Dict[str, SupplierQuote],
    step: int = 5,
    objective: str = "hybrid",
    esg_weight: float = 0.3,
    carbon_weight: float = 0.3,
    risk_weight: float = 0.4,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    suppliers = SUPPLIERS
    max_share = scenario.max_supplier_share
    min_non_rus = scenario.min_non_russia_share

    landed = {s: quotes[s].landed_cost for s in suppliers}
    carbon = {s: quotes[s].carbon_intensity for s in suppliers}
    esg = {s: quotes[s].esg for s in suppliers}

    best_score = float("inf"); best_mix: Optional[Dict[str,int]] = None
    steps = list(range(0, 101, step))

    def recurse(idx, remaining, current):
        nonlocal best_score, best_mix
        if idx == len(suppliers)-1:
            s = suppliers[idx]; share = remaining
            if share < 0 or share > 100: return
            if share/100.0 > max_share: return
            current[s] = share
            # Constraint: non-Russia minimum
            non_russia_pct = 100 - current.get("Russia", 0)
            if non_russia_pct/100.0 < min_non_rus: return

            w_cost   = sum((current[x]/100.0)*landed[x] for x in suppliers)
            w_esgGap = sum((current[x]/100.0)*(100-esg[x]) for x in suppliers)
            w_carbon = sum((current[x]/100.0)*carbon[x] for x in suppliers)

            if objective == "cost":
                score = w_cost
            elif objective == "risk":
                score = 0.5*w_esgGap + 0.5*(w_carbon/50.0)
            else:
                score = (risk_weight*(0.5*w_esgGap + 0.5*(w_carbon/50.0))
                         + esg_weight*(w_esgGap)
                         + carbon_weight*(w_carbon/50.0)
                         + (1-risk_weight-esg_weight-carbon_weight)*w_cost)

            if score < best_score:
                best_score = score; best_mix = current.copy()
            return

        s = suppliers[idx]
        for sh in steps:
            if sh/100.0 > max_share: continue
            if sh > remaining: continue
            current[s] = sh
            recurse(idx+1, remaining - sh, current)
        if s in current: del current[s]

    recurse(0, 100, {})
    shares = {k: v/100.0 for k,v in (best_mix or {}).items()}
    if not shares:
        # equal fallback under cap
        n = len(suppliers); eq = min(scenario.max_supplier_share, 1.0/n)
        shares = {s: eq for s in suppliers}
        total = sum(shares.values())
        shares = {k: v/total for k,v in shares.items()}

    w_cost   = sum(shares[x]*landed[x] for x in suppliers)
    w_carbon = sum(shares[x]*carbon[x] for x in suppliers)
    w_esg    = sum(shares[x]*esg[x]    for x in suppliers)

    return shares, {
        "weighted_landed_cost_usd_bbl": w_cost,
        "weighted_carbon_kgco2e_bbl": w_carbon,
        "weighted_esg": w_esg,
    }

# ------------------------------
# GenAI Advisory
# ------------------------------

def _template_advisory(scenario: Scenario, shares: Dict[str, float], metrics: Dict[str,float]) -> str:
    top = sorted(shares.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ", ".join(f"{k}: {v*100:.1f}%" for k,v in top)
    return (f"Recommended blend prioritizes {top_str}. "
            f"Weighted landed cost ≈ ${metrics['weighted_landed_cost_usd_bbl']:.2f}/bbl; "
            f"avg ESG ≈ {metrics['weighted_esg']:.0f}, carbon ≈ {metrics['weighted_carbon_kgco2e_bbl']:.1f} kgCO2e/bbl. "
            f"Maintain non-Russia share ≥ {scenario.min_non_russia_share:.0%} "
            f"to manage tariff exposure ({scenario.tariff_rate:.0%}).")

def genai_advisory(
    scenario: Scenario, shares: Dict[str, float], quotes: Dict[str,SupplierQuote],
    metrics: Dict[str,float], sentiment: float
) -> Tuple[str, bool]:
    if not OPENAI_ENABLED or _client is None:
        return _template_advisory(scenario, shares, metrics), False

    try:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        mix_sorted = sorted(shares.items(), key=lambda x: x[1], reverse=True)
        supplier_lines = []
        for s, p in mix_sorted:
            q = quotes[s]
            supplier_lines.append(
                f"{s}: share={p:.3f}, landed=${q.landed_cost:.2f}/bbl, carbon={q.carbon_intensity}, ESG={q.esg}"
            )
        prompt = f"""
You are an expert energy trade advisor for Government of India.
Context UTC: {now_iso()}.

Scenario:
- Tariff surcharge proxy on crude: {scenario.tariff_rate:.2%}
- Demand: {scenario.demand_bpd:,} bbl/day
- Max supplier share: {scenario.max_supplier_share:.0%}
- Minimum non-Russia share: {scenario.min_non_russia_share:.0%}
- Destination port: {scenario.selected_indian_port}
- Market sentiment (−1..+1): {sentiment:+.2f}

Supplier metrics:
{chr(10).join(supplier_lines)}

Weighted metrics:
- Landed cost: ${metrics['weighted_landed_cost_usd_bbl']:.2f}/bbl
- Carbon: {metrics['weighted_carbon_kgco2e_bbl']:.1f} kgCO2e/bbl
- ESG: {metrics['weighted_esg']:.0f}

Task:
1) Recommend a revised supplier blend (percentages sum to 100%).
2) Explain rationale (cost vs risk vs carbon) in 3 bullets.
3) One negotiation tip and one logistics tip.
Respond in 100–140 words, concise and executive-friendly.
"""
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a concise, data-grounded policy advisor."},
                {"role":"user","content":prompt}
            ],
            temperature=0.4,
            max_tokens=260,
        )
        text = resp.choices[0].message.content.strip()
        return text, True
    except Exception:
        return _template_advisory(scenario, shares, metrics), False

# ------------------------------
# Orchestrator
# ------------------------------

def run_full_pipeline(
    tariff_rate: float,
    demand_bpd: int,
    max_supplier_share: float,
    min_non_russia_share: float,
    selected_indian_port: str,
    optimizer_objective: str = "hybrid",
):
    live_bench = fetch_live_benchmarks()
    sc = Scenario(
        tariff_rate=tariff_rate,
        demand_bpd=demand_bpd,
        max_supplier_share=max_supplier_share,
        min_non_russia_share=min_non_russia_share,
        selected_indian_port=selected_indian_port,
    )
    quotes = scenario_quotes(sc, live_bench)
    shares, metrics = optimize_mix(sc, quotes, objective=optimizer_objective)
    news_sentiment, headlines = sentiment_from_feeds()
    advisory, used_llm = genai_advisory(sc, shares, quotes, metrics, news_sentiment)
    return {
        "scenario": sc,
        "live_benchmarks": live_bench,
        "quotes": quotes,
        "mix": shares,
        "metrics": metrics,
        "news_sentiment": news_sentiment,
        "headlines": headlines,
        "advisory": advisory,
        "advisory_used_llm": used_llm,
    }

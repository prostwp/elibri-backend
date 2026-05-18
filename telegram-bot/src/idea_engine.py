"""Trade-idea engine — rule-based, deterministic, always returns a Setup.

Used by every analyzer in `analyzers.py` to produce the 🎯 ИДЕЯ block.

Design:
- ALWAYS returns a Setup (low-confidence is OK — the confidence number tells
  the author whether to trust it). No "no signal" branch.
- Confidence is a 0-100 score on a transparent checklist (sum of weighted
  contributions: trend match, RSI extremum, MACD cross, level proximity,
  regime alignment). Each rule is documented inline so we can audit.
- Direction picked from the dominant signals (RSI extreme + trend), not
  from a learned model — this is the rule-based "Phase 1" baseline. The
  ML pattern-match enrichment is added later (Phase 6) and merely *adjusts*
  the confidence, never replaces the direction.
- Entry/SL/TP from ATR + nearest support/resistance. Symmetric ATR multiples
  (1.5/1.5 default) — same lesson learned from Scenario A's asymmetric-barrier
  bias collapse.

Tests live in tests/test_idea_engine.py and were written FIRST (TDD per
`feedback_ml_pipeline_safety.md`). Don't expand this file beyond what
tests cover.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

Direction = Literal["long", "short"]
Trend = Literal["uptrend", "downtrend", "sideways"]
Regime = Literal["bull", "chop", "bear", "warm"]


@dataclass
class SignalContext:
    """Snapshot of the current bar's TA + regime context.

    All fields plain Python — no pandas/numpy in the dataclass so unit tests
    stay fast and don't require parquet fixtures.
    """
    rsi: float                   # 0-100
    macd_line: float
    macd_signal: float
    trend: Trend                 # heuristic from `_trend()` in analyzers.py
    price: float                 # latest close
    support: float               # nearest 50-bar low
    resistance: float            # nearest 50-bar high
    atr: float                   # 14-period ATR (price units)
    regime: Regime = "chop"      # 1d EMA-200 slope classification


def infer_regime(closes_1d: list[float]) -> tuple[Regime, float]:
    """Coarse regime classifier — returns (label, raw_slope_clipped).

    Bug #2 fix (review 2026-04-30): now returns the raw slope value too, so
    pattern_match.compute_query_features can pass it as the `regime_score`
    feature instead of the hardcoded 0.0 placeholder (which was biasing
    HNSW retrieval toward bear/chop bars after z-normalization).

    Mirrors `regime_classifier.compute_regime_score` from ml-training but
    self-contained (no pandas/numpy dep). Threshold ±0.04 matches the build
    script's `classify_regime_per_bar` — keep in sync.
    """
    n = len(closes_1d)
    if n < 250:
        return "warm", 0.0

    # Lightweight EMA200 (running EMA in pure Python).
    k = 2 / 201
    ema = closes_1d[0]
    ema_history: list[float] = []
    for c in closes_1d:
        ema = c * k + ema * (1 - k)
        ema_history.append(ema)
    ema_now = ema_history[-1]
    ema_then = ema_history[-50] if n >= 50 else ema_history[0]
    # ATR proxy: average abs(close[i]-close[i-1]) over last 14 bars.
    atr_proxy = sum(abs(closes_1d[-i] - closes_1d[-i - 1]) for i in range(1, 15)) / 14
    if atr_proxy < 1e-9:
        return "chop", 0.0
    slope = (ema_now - ema_then) / (50 * atr_proxy)
    slope_clipped = max(-5.0, min(5.0, slope))
    if slope >= 0.04:
        return "bull", slope_clipped
    if slope <= -0.04:
        return "bear", slope_clipped
    return "chop", slope_clipped


@dataclass
class Setup:
    """Trade idea + transparent confidence + reasons.

    `reasons` is a human-readable list ('RSI=27 → oversold', 'price near
    support', ...) used by the analysis-section template to explain WHY
    the direction was picked. Limit ~5 reasons for readability.
    """
    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    confidence: int              # 0-100
    reasons: List[str] = field(default_factory=list)

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.take_profit - self.entry)
        if risk < 1e-9:
            return 0.0
        return reward / risk


# ─── Direction picker ────────────────────────────────────────────────────────

def pick_direction(ctx: SignalContext) -> Direction:
    """Pick LONG or SHORT based on dominant signals.

    Priority:
    1. RSI extremum overrides everything (oversold→long, overbought→short).
    2. Trend alignment (uptrend→long, downtrend→short).
    3. MACD cross.
    4. Default: long bias on chop (slight optimistic prior — historically
       crypto more often goes up than down on 4h horizons; we explicitly
       penalise this in confidence so the score reflects the weakness).
    """
    if ctx.rsi <= 30:
        return "long"
    if ctx.rsi >= 70:
        return "short"
    if ctx.trend == "uptrend":
        return "long"
    if ctx.trend == "downtrend":
        return "short"
    if ctx.macd_line > ctx.macd_signal:
        return "long"
    if ctx.macd_line < ctx.macd_signal:
        return "short"
    return "long"


# ─── Confidence scorer ───────────────────────────────────────────────────────

def score_confidence(ctx: SignalContext, direction: Direction) -> int:
    """Confidence on 0-100 scale via additive checklist.

    Components (max contribution in parens — sum to 100):
    - RSI alignment with direction (25)
    - Trend alignment (20)
    - MACD cross alignment (15)
    - Level proximity (price near support for long / resistance for short) (15)
    - Regime alignment (15)
    - R/R structurally sane (entry, sl, tp from ATR ≥ 1.0 R/R) (10)
    """
    score = 0.0

    # RSI (25 max). Extremes favourable to direction.
    if direction == "long":
        if ctx.rsi <= 30:
            score += 25
        elif ctx.rsi <= 40:
            score += 15
        elif ctx.rsi <= 50:
            score += 8
        elif ctx.rsi >= 70:
            score -= 10  # against direction
    else:  # short
        if ctx.rsi >= 70:
            score += 25
        elif ctx.rsi >= 60:
            score += 15
        elif ctx.rsi >= 50:
            score += 8
        elif ctx.rsi <= 30:
            score -= 10

    # Trend (20 max).
    if direction == "long" and ctx.trend == "uptrend":
        score += 20
    elif direction == "short" and ctx.trend == "downtrend":
        score += 20
    elif ctx.trend == "sideways":
        score += 5
    else:  # against trend
        score -= 5

    # MACD (15 max).
    macd_bull = ctx.macd_line > ctx.macd_signal
    if direction == "long" and macd_bull:
        score += 15
    elif direction == "short" and not macd_bull:
        score += 15
    else:
        score -= 3

    # Level proximity (15 max). Helpful: long near support, short near resistance.
    width = ctx.resistance - ctx.support
    if width > 0:
        if direction == "long":
            dist_pct = (ctx.price - ctx.support) / width
            # 0.0 = at support (best), 1.0 = at resistance (worst for long entry)
            if dist_pct <= 0.2:
                score += 15
            elif dist_pct <= 0.4:
                score += 8
            elif dist_pct >= 0.8:
                score -= 5
        else:  # short
            dist_pct = (ctx.resistance - ctx.price) / width
            if dist_pct <= 0.2:
                score += 15
            elif dist_pct <= 0.4:
                score += 8
            elif dist_pct >= 0.8:
                score -= 5

    # Regime alignment (15 max). Long in bull / short in bear is rewarded.
    if direction == "long":
        if ctx.regime == "bull":
            score += 15
        elif ctx.regime == "chop":
            score += 5
        elif ctx.regime == "bear":
            score -= 10
    else:  # short
        if ctx.regime == "bear":
            score += 15
        elif ctx.regime == "chop":
            score += 5
        elif ctx.regime == "bull":
            score -= 10

    # ATR sanity (10 max): if ATR is meaningful and we have room between
    # support/resistance for a 1.5 ATR target, this is a tradeable bar.
    if ctx.atr > 0 and width > 1.5 * ctx.atr:
        score += 10

    return int(max(0, min(100, round(score))))


# ─── Setup builder ───────────────────────────────────────────────────────────

def build_setup(ctx: SignalContext, atr_mult_sl: float = 1.5,
                atr_mult_tp: float = 2.0) -> Setup:
    """Always returns a Setup. Caller decides whether to publish based on
    `confidence` field.

    Default 1.5 ATR stop / 2.0 ATR target → R/R ≈ 1.33. ATR-symmetric (no
    asymmetric prior bias — Scenario A lesson). For a tighter "scalper"
    style override mults via call-site config.

    Inverted-levels (support>resistance — temporary feed glitch) handled by
    swapping for the math; no exception raised.
    """
    direction = pick_direction(ctx)
    confidence = score_confidence(ctx, direction)
    reasons = _explain(ctx, direction)

    # Defensive: if support/resistance are inverted, swap for the math.
    sup = min(ctx.support, ctx.resistance)
    res = max(ctx.support, ctx.resistance)

    # ATR fallback when measurement is degenerate (0 or NaN-like).
    atr = ctx.atr if ctx.atr > 0 else max((res - sup) * 0.05, ctx.price * 0.005)

    entry = ctx.price
    if direction == "long":
        sl_atr = entry - atr_mult_sl * atr
        sl = max(sl_atr, sup * 0.999)  # prefer support, ATR is hard floor
        # If support already above entry (rare with bad data), use ATR only.
        if sl >= entry:
            sl = entry - atr_mult_sl * atr
        tp = entry + atr_mult_tp * atr
        # Don't aim past resistance unrealistically.
        tp = min(tp, res * 1.001) if res > entry else entry + atr_mult_tp * atr
        if tp <= entry:
            tp = entry + atr_mult_tp * atr
    else:  # short
        sl_atr = entry + atr_mult_sl * atr
        sl = min(sl_atr, res * 1.001)
        if sl <= entry:
            sl = entry + atr_mult_sl * atr
        tp = entry - atr_mult_tp * atr
        tp = max(tp, sup * 0.999) if sup < entry else entry - atr_mult_tp * atr
        if tp >= entry:
            tp = entry - atr_mult_tp * atr

    return Setup(
        direction=direction,
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        confidence=confidence,
        reasons=reasons,
    )


# ─── Reasoning text (consumed by templates.py later) ─────────────────────────

def _explain(ctx: SignalContext, direction: Direction) -> List[str]:
    """Top 3-5 reasons to mention in the analysis block. Order = strength."""
    reasons: List[str] = []

    if direction == "long":
        if ctx.rsi <= 30:
            reasons.append(f"RSI={ctx.rsi:.0f} — перепроданность")
        elif ctx.rsi <= 45:
            reasons.append(f"RSI={ctx.rsi:.0f} — близко к зоне покупок")
        if ctx.trend == "uptrend":
            reasons.append("восходящий тренд по EMA")
        if ctx.macd_line > ctx.macd_signal:
            reasons.append("MACD: бычье пересечение")
        if ctx.regime == "bull":
            reasons.append("глобальный режим — bull")
        # Level proximity
        width = ctx.resistance - ctx.support
        if width > 0 and (ctx.price - ctx.support) / width <= 0.25:
            reasons.append(f"цена у поддержки {ctx.support:.0f}")
    else:  # short
        if ctx.rsi >= 70:
            reasons.append(f"RSI={ctx.rsi:.0f} — перекупленность")
        elif ctx.rsi >= 55:
            reasons.append(f"RSI={ctx.rsi:.0f} — близко к зоне продаж")
        if ctx.trend == "downtrend":
            reasons.append("нисходящий тренд по EMA")
        if ctx.macd_line < ctx.macd_signal:
            reasons.append("MACD: медвежье пересечение")
        if ctx.regime == "bear":
            reasons.append("глобальный режим — bear")
        width = ctx.resistance - ctx.support
        if width > 0 and (ctx.resistance - ctx.price) / width <= 0.25:
            reasons.append(f"цена у сопротивления {ctx.resistance:.0f}")

    if not reasons:
        reasons.append("сильных сигналов нет — слабый setup")
    return reasons[:5]

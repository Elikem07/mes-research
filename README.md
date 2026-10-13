# MES Intraday Reaction Research

A measurement framework for intraday MES (Micro E-mini S&P 500) price behavior.
It deliberately stops short of designing a strategy: the question it answers is
what price *actually does* at a level, conditioned on regime, before any entry
or exit rule exists.

> **Research code.** Not a trading system and not investment advice.

## The research question

Before building a strategy, classify every level touch into one of five
reaction outcomes and estimate

```
P(outcome | regime)
```

across 16 regime variables. If the conditional distribution is
indistinguishable from the unconditional one, there is nothing to build a
strategy on — and that is a result worth having cheaply, before committing to
an execution stack.

Because the framework contains no entry or exit rules, there were none to
redact when preparing this repository for publication. The only externalized
values are broker cost assumptions (see [Configuration](#configuration)).

## Structure

| File | Role |
|---|---|
| `mes_research_project.py` | Foundation: cost model (three tiers), `Signal` / `Trade` types, P&L accounting |
| `experiments/exp1_taxonomy.py` | Experiment 1 — the reaction taxonomy and regime conditioning |
| `fetch_mes_data.py` | Historical bar fetch and caching |

## Method

- **Three cost tiers.** Every result is reported under optimistic, base and
  stress friction. A finding that only survives the optimistic tier is not a
  finding.
- **Point-in-time features.** Bars carry an `available_at` timestamp offset from
  their close, so a feature cannot be used before it existed. This is the
  defense against lookahead bias.
- **Multiple-comparisons control.** Regime reports apply Benjamini–Hochberg FDR
  correction (q=0.05) and Wilson confidence intervals for proportions, with a
  minimum cell count before a bucket is reported at all.
- **Repeat-test tracking.** First touches of a level are distinguished from
  repeat touches, since they are not the same event.

Statistical settings (horizon, warmup, FDR q, minimum n, CI z) remain in code
rather than config — they are methodology, not tunable parameters, and
publishing them is the point.

## Install

```bash
pip install pandas numpy
```

Python 3.11+ (`tomllib`). Fetching data requires a futures data source with
intraday history.

## Configuration

Broker friction is account-specific and gitignored:

```bash
cp config/costs.example.toml config/costs.toml
# then edit config/costs.toml with your commission schedule
```

| File | Committed? | Contents |
|---|---|---|
| `config/costs.example.toml` | yes | Schema; contract specs real, friction zeroed |
| `config/costs.toml` | **no** | Your real commissions and slippage estimates |

Override with `MES_COSTS_CONFIG`. The example sets all friction to zero, which
is **not** a realistic assumption — it will overstate every result, and loading
it emits a `RuntimeWarning` saying so.

`[contract]` holds MES reference data (tick value $1.25, multiplier $5.00/point).
`[tier.optimistic|base|stress]` hold commission and slippage per tier.

## Usage

```bash
python fetch_mes_data.py                  # cache bars to ./data/
python mes_research_project.py --validate  # cost-model and engine self-test
python experiments/exp1_taxonomy.py        # reaction taxonomy by regime
```

Cached data is gitignored.

## Limitations

- Reaction outcomes are a chosen taxonomy; a different classification could
  reach different conclusions from the same bars.
- Regime variables are correlated, so "16 regimes" is fewer than 16 independent
  tests — the FDR correction helps but does not fully resolve this.
- Cost tiers are estimates. Real fills in fast markets can exceed the stress
  tier.
- Measuring a conditional distribution is not the same as establishing a
  tradeable edge. Nothing here has been validated out-of-sample as a strategy,
  because nothing here is a strategy.

## License

No license granted. All rights reserved.

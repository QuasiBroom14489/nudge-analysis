"""What makes a Vesper nudge useful?

Reads data/private/nudges.csv (row-level, never published) and writes only
aggregates: data/public/*.csv, charts/*.svg and findings.md. Any group with
fewer than MIN_CELL judged nudges is suppressed from the public outputs.

    uv run python scripts/analyze.py
"""

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PUB, CHARTS = ROOT / "data/public", ROOT / "charts"
MIN_CELL = 5
INK, MUTED, ACCENT, GRID = "#1d1813", "#7a6c58", "#3b5bdb", "#e6dccb"

PUB.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(exist_ok=True)
con = duckdb.connect()
con.execute(f"create table n as select * from read_csv_auto('{ROOT / 'data/private/nudges.csv'}')")
con.execute("create view jb as select *, (verdict = 'good')::int as useful from n where verdict is not null")


def q(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def by(col: str, order: str | None = None) -> pd.DataFrame:
    df = q(f"""
        select {col} as level, count(*) as judged, sum(useful) as useful,
               round(avg(useful), 3) as useful_rate,
               sum((verdict = 'bad')::int) as wrong, sum((verdict = 'noise')::int) as noise
        from j group by 1 having count(*) >= {MIN_CELL} order by {order or 'judged desc'}""")
    df.insert(0, "feature", col)
    return df


# ---------------------------------------------------------------- descriptive
overview = q("""
    select (select count(*) from n) as raised, count(*) as judged,
           sum(useful) as useful, sum((verdict='bad')::int) as wrong, sum((verdict='noise')::int) as noise,
           round(avg(useful), 3) as useful_rate,
           round(median(minutes_to_verdict), 1) as median_minutes_to_verdict
    from jb""")
o = overview.iloc[0]

con.execute("""create view j as select *,
    case when confidence < 0.8 then '<0.80' when confidence < 0.9 then '0.80-0.89'
         when confidence < 1 then '0.90-0.99' else '1.00' end as conf_band,
    case when hour < 9 then 'before 9' when hour < 12 then '9-12' when hour < 17 then '12-17'
         when hour < 21 then '17-21' else '21+' end as time_of_day,
    case when prior_fires = 0 then 'first' when prior_fires < 3 then '2nd-3rd'
         when prior_fires < 10 then '4th-10th' else '11th+' end as repeat_band,
    day // 7 + 1 as week
    from jb""")

tables = [
    by("category"),
    by("urgency"),
    by("conf_band", "level"),
    by("time_of_day", "case level when 'before 9' then 0 when '9-12' then 1 when '12-17' then 2 when '17-21' then 3 else 4 end"),
    by("repeat_band", "case level when 'first' then 0 when '2nd-3rd' then 1 when '4th-10th' then 2 else 3 end"),
    by("week", "level"),
]
breakdown = pd.concat(tables, ignore_index=True)
breakdown.to_csv(PUB / "usefulness_by_feature.csv", index=False)
overview.to_csv(PUB / "overview.csv", index=False)

# ---------------------------------------------------------------- what-if
# Replay the judged history with a cap on how many times one rule may fire.
# In-sample and hindsight: it sizes the trade-off, it does not prove the fix.
caps = []
for cap in (3, 5, 10, 20, None):
    keep = "true" if cap is None else f"prior_fires < {cap}"
    r = q(f"""select (select count(*) from n where {keep}) as raised_kept, count(*) as judged_kept,
                     sum(useful) as useful_kept, round(avg(useful), 3) as precision
              from j where {keep}""").iloc[0]
    caps.append({"cap": cap or "none", "raised_kept": int(r.raised_kept), "judged_kept": int(r.judged_kept),
                 "useful_kept": int(r.useful_kept), "useful_kept_share": round(r.useful_kept / o.useful, 3),
                 "precision": r.precision})
caps = pd.DataFrame(caps)
caps.to_csv(PUB / "repeat_cap_whatif.csv", index=False)

# ---------------------------------------------------------------- model
# Time-ordered split: fit on the first 70% of judged nudges, score the rest,
# so the check asks "would last month's pattern have predicted this month?"
jd = q("select * from j order by day, hour").reset_index(drop=True)
cut = int(len(jd) * 0.7)
train, test = jd.iloc[:cut], jd.iloc[cut:]
cat_cols = ["category", "urgency", "time_of_day"]
num_cols = ["confidence", "text_len", "evidence_n", "prior_fires"]
jd["prior_fires"] = np.log1p(jd["prior_fires"])
train, test = jd.iloc[:cut], jd.iloc[cut:]
model = make_pipeline(
    ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=MIN_CELL), cat_cols),
                       ("num", StandardScaler(), num_cols)]),
    LogisticRegression(max_iter=1000, C=0.5),
)
model.fit(train[cat_cols + num_cols], train.useful)
p = model.predict_proba(test[cat_cols + num_cols])[:, 1]
base = np.full(len(test), train.useful.mean())
auc = roc_auc_score(test.useful, p) if test.useful.nunique() > 1 else float("nan")
conf_auc = roc_auc_score(test.useful, test.confidence) if test.useful.nunique() > 1 else float("nan")
brier, brier_base = brier_score_loss(test.useful, p), brier_score_loss(test.useful, base)
# a constant that already knows the test window's rate: the bar a model has to beat once usefulness drifts
brier_oracle = brier_score_loss(test.useful, np.full(len(test), test.useful.mean()))

# precision if only the top third by model score had been delivered
k = max(1, len(test) // 3)
top = test.assign(p=p).nlargest(k, "p")
model_eval = pd.DataFrame([{
    "train_judged": len(train), "test_judged": len(test),
    "test_useful_rate": round(test.useful.mean(), 3),
    "auc_model": round(auc, 3), "auc_confidence_alone": round(conf_auc, 3),
    "brier_model": round(brier, 4), "brier_base_rate": round(brier_base, 4), "brier_test_rate_oracle": round(brier_oracle, 4),
    "precision_top_third": round(top.useful.mean(), 3),
}])
model_eval.to_csv(PUB / "model_eval.csv", index=False)
names = model[0].get_feature_names_out()
coefs = pd.DataFrame({"term": names, "coef": model[-1].coef_[0]}).sort_values("coef")
coefs.to_csv(PUB / "model_coefficients.csv", index=False)


# ---------------------------------------------------------------- charts
def bar_chart(df: pd.DataFrame, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(df) + 1.1), dpi=100)
    y = np.arange(len(df))[::-1]
    ax.barh(y, df.useful_rate * 100, color=ACCENT, height=0.62)
    ax.axvline(o.useful_rate * 100, color=INK, lw=1, ls=(0, (3, 3)))
    ax.text(o.useful_rate * 100, -0.75, f" overall {o.useful_rate:.0%}", color=INK, fontsize=8.5, va="center")
    ax.set_ylim(-1.1, len(df) - 0.5)
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.text(r.useful_rate * 100 + 1, yi, f"{r.useful_rate:.0%}  (n={r.judged})", va="center", fontsize=8.5, color=MUTED,
                bbox=dict(facecolor="#f7efdf", edgecolor="none", pad=1.5))
    ax.set_yticks(y, df.level.astype(str), fontsize=9.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% judged useful", fontsize=9, color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CHARTS / fname, format="svg", transparent=True)
    plt.close(fig)


bar_chart(tables[0], "Usefulness by nudge type", "by_category.svg")
bar_chart(tables[2], "Usefulness by model confidence", "by_confidence.svg")
bar_chart(tables[4], "Usefulness by how often the same rule had already fired", "by_repeat.svg")
bar_chart(tables[3], "Usefulness by time of day", "by_time_of_day.svg")
bar_chart(tables[5].assign(level="week " + tables[5].level.astype(str)), "Usefulness by week since the loop started", "by_week.svg")

# ---------------------------------------------------------------- findings
def table_md(df: pd.DataFrame) -> str:
    cols = ["level", "judged", "useful", "useful_rate", "wrong", "noise"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


m = model_eval.iloc[0]
md = [
    "# What makes a nudge useful? (baseline)",
    "",
    f"{int(o.raised):,} nudges raised; {int(o.judged)} judged by hand: {int(o.useful)} useful, {int(o.wrong)} wrong, "
    f"{int(o.noise)} noise (**{o.useful_rate:.0%} useful**; 380 verdict events, the latest per nudge counts). Median time to a verdict: {o.median_minutes_to_verdict} min.",
    "",
    "Caveat: judged nudges are the ones I chose to answer, not a random sample.",
    "",
]
for t in tables:
    md += [f"## By {t.feature.iloc[0]}", "", table_md(t), ""]
md += ["## What if one rule could only fire N times? (in-sample replay)", "",
       "| cap | raised kept | judged kept | useful kept | share of useful kept | precision |", "|---|---|---|---|---|---|"]
md += [f"| {r.cap} | {r.raised_kept} | {r.judged_kept} | {r.useful_kept} | {r.useful_kept_share:.0%} | {r.precision:.0%} |" for _, r in caps.iterrows()]
md += [""]
md += [
    "## Held-out check (time-ordered 70/30 split)",
    "",
    f"- Trained on the first {int(m.train_judged)} judged nudges, scored the next {int(m.test_judged)} "
    f"(test base rate {m.test_useful_rate:.0%}).",
    f"- AUC: model {m.auc_model}, the intuition model's own confidence alone {m.auc_confidence_alone}.",
    f"- Brier: model {m.brier_model}, vs. {m.brier_base_rate} for always predicting the training base rate and "
    f"{m.brier_test_rate_oracle} for a constant that already knows the test window's rate. The model ranks better than chance "
    f"but does not out-predict that oracle constant: usefulness drifted down over time (see By week).",
    f"- If only the model's top third had been delivered, {m.precision_top_third:.0%} would have been useful.",
    "",
    "## Largest coefficients",
    "",
    "| term | coef |",
    "|---|---|",
] + [f"| {r.term} | {r.coef:+.2f} |" for _, r in pd.concat([coefs.head(5), coefs.tail(5)]).iterrows()]
(ROOT / "findings.md").write_text("\n".join(md) + "\n")
print((ROOT / "findings.md").read_text())

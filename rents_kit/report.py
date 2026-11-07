"""report.html in the sample report's layout (US10, T10.1 to T10.7). Aggregates only (T10.5).
Reads out/final_meta.json, out/index.csv, out/eval_*.json and out/results.jsonl written by eval/final."""
import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from features import ANCHOR_NAMES  # noqa: E402

CSS = """body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:32px auto;color:#1d2b3a;line-height:1.5}
h1{font-size:26px}h2{font-size:20px;margin-top:34px;border-bottom:2px solid #e3e8ef;padding-bottom:4px}
.tiles{display:flex;gap:14px;flex-wrap:wrap}.tile{flex:1;min-width:190px;background:#f3f6fa;border-radius:8px;padding:12px}
.tile b{font-size:24px;display:block}table{border-collapse:collapse;width:100%;font-size:14px}
td,th{border-bottom:1px solid #e3e8ef;padding:5px 8px;text-align:left}img{max-width:100%}.note{color:#5b6b7c;font-size:13px}"""


def _img(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}">'


def _bar(labels, values, title, ylabel):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar([str(x) for x in labels], values, color="#2f6db5")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    return _img(fig)


def _resid(A, d):
    """Recent adverts, log rent relative to suburb x bedroom median (removes location and size)."""
    a = A[A["in_input"] & (A["advert_date"] >= pd.Timestamp(C.CUTOFF) - pd.DateOffset(months=24))].copy()
    a["lr"] = np.log(a["rent"])
    a["rel"] = a["lr"] - a.groupby(["suburb_clean", "bed_group"])["lr"].transform("median")
    f = d.set_index("gnaf_id")
    for c in ("floors_above_nbrs", "new_home_share"):
        a[c] = a["gnaf_id"].map(f[c])
    return a


def insights(A, d, meta):
    a = _resid(A, d)
    med = float(a["rent"].median())
    out = []
    # 1 rent per extra floor (within building and bedroom group)
    b = a.dropna(subset=["floor"])
    b = b[b.groupby("bkey")["floor"].transform("nunique") > 1]
    if len(b) > 200:
        x = b["floor"] - b.groupby(["bkey", "bed_group"])["floor"].transform("mean")
        y = b["lr"] - b.groupby(["bkey", "bed_group"])["lr"].transform("mean")
        slope = float((x * y).sum() / (x * x).sum())
        out.append(("Each extra floor", f"${med * slope:+.0f} a week",
                    f"Within the same building and bedroom count, from {len(b):,} adverts in the last 24 months "
                    f"(median rent ${med:.0f})."))
    # 2 parking premium within building
    prem = meta.get("premium_log_by_bed", {})
    if prem:
        rows = [(k, med * (np.exp(v) - 1)) for k, v in prem.items() if k not in ("overall", "n_buildings")]
        ov = med * (np.exp(prem.get("overall", 0)) - 1)
        out.append(("Parking within the same building", f"${ov:.0f} a week",
                    f"Compared like for like inside {int(prem.get('n_buildings', 0)):,} buildings; the sample report's "
                    f"suburb-level gap was $90. By bedroom group: " +
                    ", ".join(f"{'studio' if float(k) == 0 else ('4+' if float(k) >= 4 else int(float(k)))}: ${v:.0f}" for k, v in sorted(rows, key=lambda r: float(r[0])))))
    # 3 floors above neighbours
    v = a.dropna(subset=["floors_above_nbrs"])
    if len(v) > 200:
        band = pd.cut(v["floors_above_nbrs"], [-99, -1, 4, 99], labels=["below neighbours", "0-4 floors above", "5+ floors above"])
        g = (np.exp(v.groupby(band, observed=True)["rel"].median()) - 1) * med
        out.append(("Above the neighbours (views proxy)", f"${g.iloc[-1] - g.iloc[0]:+.0f} a week",
                    "Flats 5+ floors above the tallest building within 100 m versus flats below their neighbours' tops, "
                    "same suburb and bedrooms.", _bar(g.index, g.values, "Rent vs suburb-bedroom median", "$ a week")))
    # 4 new supply
    s = a.dropna(subset=["new_home_share"])
    if len(s) > 200:
        band = pd.cut(s["new_home_share"], [-0.01, 0.05, 0.15, 10], labels=["<5% new", "5-15% new", ">15% new"])
        g = (np.exp(s.groupby(band, observed=True)["rel"].median()) - 1) * med
        out.append(("New supply in the block", f"${g.iloc[-1] - g.iloc[0]:+.0f} a week",
                    "Blocks where more than 15% of homes were added 2021-2026 versus under 5%, same suburb and bedrooms.",
                    _bar(g.index, g.values, "Rent vs suburb-bedroom median by new-home share", "$ a week")))
    return out


def make_report(d, A, info, ainfo, path=None):
    out = C.OUT_DIR
    meta = json.loads((out / "final_meta.json").read_text(encoding="utf-8")) if (out / "final_meta.json").exists() else {}
    evals = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("eval_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)}
    tune = next((v for k, v in evals.items() if "tune" in k), None)          # most recent run of each
    conf = next((v for k, v in evals.items() if "confirm" in k), None)
    best = conf or tune
    h = [f"<html><head><meta charset='utf-8'><title>What will this flat rent for?</title><style>{CSS}</style></head><body>"]
    if best:
        m = best["model"]["all"]
        h.append(f"<h1>Our estimate misses a typical City of Sydney asking rent by ${m['mdae']:.0f} a week</h1>"
                 f"<p>Reference points: simple estimate about $100, with parking about $93, Microburbs' rental valuation $86. "
                 f"{m['within100']:.1%} of {m['n']:,} later adverts were within $100.</p>")
    else:
        h.append("<h1>What will this flat rent for?</h1>")
    h.append(f"<p><b>Our clever step.</b> {html.escape(C.CLEVER_STEP)}</p>")
    h.append("<h2>What we found along the way</h2><div class='tiles'>")
    ins = insights(A, d, meta)
    for it in ins:
        h.append(f"<div class='tile'>{html.escape(it[0])}<b>{html.escape(it[1])}</b><span class='note'>{html.escape(it[2])}</span></div>")
    h.append("</div>")
    for it in ins:
        if len(it) > 3:
            h.append(it[3])
    h.append("<h2>What our separate check showed</h2>")
    if best:
        rows = "".join(f"<tr><td>{n}</td><td>${best[n]['all']['mdae']:.0f}</td><td>{best[n]['all']['within100']:.1%}</td>"
                       f"<td>{best[n]['all']['r2']:.3f}</td><td>{best[n]['all']['bias']:+.0f}</td><td>${best[n]['one_per_campaign']['mdae']:.0f}</td></tr>"
                       for n in ("model", "own last rent x index", "sample method", "suburb x bedrooms", "suburb median"))
        h.append(f"<p>Built on adverts before {best['train_end']}; tested on adverts {best['test'][0]} to {best['test'][1]} "
                 f"that we never used to build or tune. Scored like the judges: the column matching the advert's parking.</p>"
                 f"<table><tr><th>Method</th><th>Typical miss</th><th>Within $100</th><th>R²</th><th>Bias</th><th>One advert per listing</th></tr>{rows}</table>"
                 f"<p class='note'>R² is the share of variation in weekly rent the estimate explains. Flats with a prior advert: "
                 f"${best['model | prior advert']['mdae']:.0f} typical miss (n {best['model | prior advert']['n']:,}); without: "
                 f"${best['model | no prior advert']['mdae']:.0f} (n {best['model | no prior advert']['n']:,}).</p>")
        if tune and conf:
            h.append(f"<p>Tuning window typical miss ${tune['model']['all']['mdae']:.0f}; confirmation window ${conf['model']['all']['mdae']:.0f}.</p>")
    h.append("<h2>What data supported the result</h2><ul>")
    h.append(f"<li>{len(d):,} flats to estimate; {ainfo.get('adverts', 0):,} adverts after matching "
             f"({', '.join(f'{k}: {v:,}' for k, v in ainfo.get('by_source', {}).items())}); "
             f"{ainfo.get('relists_collapsed', 0):,} relists collapsed into their listing.</li>")
    h.append(f"<li>Matched by {ainfo.get('match_counts', {})}. Adverts dropped for being after 31 July 2026: {info.get('guard_drops', {})}. "
             f"Input dates after the cutoff masked: {info.get('masked', {})}.</li>")
    if "merge" in ainfo:
        h.append(f"<li>Microburbs API rent history (to 31 July 2026): {ainfo['merge']}.</li>")
    api_used = "api" in ainfo.get("by_source", {})
    h.append(f"<li>Sources: Microburbs supplied files{' and Microburbs Hackathon API rent history (to 31 July 2026)' if api_used else ''}; "
             "Open G-NAF © Geoscape Australia; ABS Census © Commonwealth of Australia, CC BY 4.0"
             + ("; LiDAR 2013/2020 © State of NSW (Spatial Services) via ELVIS, CC BY; building footprints, parks and water © OpenStreetMap contributors, ODbL" + ("; short-stay counts from Inside Airbnb, CC BY 4.0" if C.USE_AIRBNB else "") if (C.USE_GNAF or C.USE_LIDAR or C.USE_AIRBNB) and (C.PUBLIC_DIR / "flat_public_features.csv").exists() else "")
             + ". Data © Microburbs; estimates are indicative.</li></ul>")
    if (out / "index.csv").exists():
        ix = pd.read_csv(out / "index.csv")
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.plot(pd.PeriodIndex(ix["month"], freq="M").to_timestamp(), ix["index_100"], color="#2f6db5")
        ax.set_title("Rent index from repeat adverts (July 2026 = 100)", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        h.append(_img(fig))
        last = ix.tail(4)["index_100"].to_numpy()
        if len(last) == 4:
            h.append(f"<p class='note'>Our City of Sydney asking-rent index changed {last[-1] / last[0] - 1:+.1%} from April to "
                     f"July 2026 (12 months: {last[-1] / ix['index_100'].iloc[-13] - 1:+.1%}); Domain reported about +4% for "
                     f"Greater Sydney units in the June quarter (published 8 July 2026), a wider area and a different measure.</p>")
    h.append("<h2>How we estimated rent</h2><p>Earlier adverts → restated in July 2026 dollars with the rent index → "
             "each flat's own last rent, its building, its block, nearby flats and its suburb by bedrooms → a gradient-boosted "
             "model that predicts the median → two columns: with and without a car space, the gap measured inside buildings.</p>")
    if meta.get("anchor_src"):
        h.append("<p>Starting point used per flat: " + ", ".join(f"{ANCHOR_NAMES[int(k)]}: {v:,}" for k, v in meta["anchor_src"].items()) + ".</p>")
    h.append("<h2>How we checked it</h2><p>We split adverts by date, built on the earlier ones and scored the later ones once. "
             "Before tuning, we rebuilt the sample report's suburb-bedroom-parking method on its own dates to check our data handling "
             "matched Microburbs'. Nothing dated after 31 July 2026 was used.</p>")
    h.append("<h2>What worked and what did not</h2>")
    if (out / "results.jsonl").exists():
        r = pd.read_json(out / "results.jsonl", lines=True)
        r = r.drop_duplicates(["name", "fold"], keep="last").sort_values("mdae")
        h.append("<table><tr><th>Run</th><th>Window</th><th>Typical miss</th><th>Within $100</th></tr>" + "".join(
            f"<tr><td>{x.name}</td><td>{x.fold}</td><td>${x.mdae:.0f}</td><td>{x.within100:.1%}</td></tr>" for x in r.itertuples()) + "</table>")
    h.append("<p>Facing direction was not attempted: no data says which side of a building a flat faces.</p>")
    h.append("<h2>What remains uncertain</h2><ul><li>Rents moved fast in 2026; the August-September adverts may sit above our July level.</li>"
             "<li>Flats with no earlier advert rely on their building and neighbours, and miss by more.</li>"
             "<li>Microburbs' advert coverage has gaps, so some histories are incomplete.</li></ul></body></html>")
    path = Path(path or out / "report.html")
    path.write_text("\n".join(h), encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    leak = [g for g in d["gnaf_id"].head(2000) if g in text]                 # T10.5 aggregates-only check
    if leak:
        raise SystemExit(f"report contains row-level ids ({leak[:3]}) - aggregates only")
    print(f"wrote {path}")
    return path

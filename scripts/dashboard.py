#!/usr/bin/env python
"""학습 메트릭(metrics.jsonl) → 자체포함 HTML 대시보드.

trainer가 out_dir/metrics.jsonl 에 스텝별 지표를 남깁니다. 이 스크립트는 그 데이터를
읽어 손실/LR/MFU/처리량/검증손실 곡선을 담은 단일 HTML 파일을 만듭니다.
(Chart.js는 CDN 로드, 데이터는 HTML에 인라인 → 오프라인에서도 파일만 열면 됨)

사용 예:
    python scripts/dashboard.py --metrics runs/10b/metrics.jsonl --out runs/10b/dashboard.html
"""
import argparse
import json
import os
import sys


def load_metrics(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None:
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


HTML = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>학습 대시보드</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  body{{font-family:system-ui,'Segoe UI',sans-serif;margin:0;background:#0f1117;color:#e6e6e6}}
  header{{padding:20px 28px;border-bottom:1px solid #222}}
  h1{{margin:0;font-size:20px}} .sub{{color:#8b93a7;font-size:13px;margin-top:4px}}
  .grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;padding:24px}}
  .card{{background:#171a23;border:1px solid #232733;border-radius:12px;padding:16px}}
  .card h2{{margin:0 0 10px;font-size:14px;color:#b9c0d0;font-weight:600}}
  .kpis{{display:flex;gap:24px;flex-wrap:wrap;padding:0 28px 8px}}
  .kpi{{background:#171a23;border:1px solid #232733;border-radius:10px;padding:12px 16px;min-width:120px}}
  .kpi .v{{font-size:22px;font-weight:700}} .kpi .l{{color:#8b93a7;font-size:12px}}
  canvas{{max-height:280px}}
</style></head>
<body>
<header><h1>10B LLM · 학습 대시보드</h1>
<div class="sub">{src} · {n}개 로그 · 최신 step {last}</div></header>
<div class="kpis">{kpis}</div>
<div class="grid">
  <div class="card"><h2>Train Loss</h2><canvas id="loss"></canvas></div>
  <div class="card"><h2>Learning Rate</h2><canvas id="lr"></canvas></div>
  <div class="card"><h2>MFU (%)</h2><canvas id="mfu"></canvas></div>
  <div class="card"><h2>Throughput (tok/s)</h2><canvas id="tps"></canvas></div>
  <div class="card"><h2>Grad Norm</h2><canvas id="gn"></canvas></div>
  <div class="card"><h2>Val Loss</h2><canvas id="val"></canvas></div>
</div>
<script>
const D = {data};
function mk(id,label,xy,color,mul){{
  if(!xy.x.length){{document.getElementById(id).parentElement.style.opacity=.4;return;}}
  new Chart(document.getElementById(id),{{type:'line',
    data:{{labels:xy.x,datasets:[{{label:label,data:xy.y.map(v=>mul?v*mul:v),
      borderColor:color,backgroundColor:color+'22',borderWidth:2,pointRadius:0,tension:.15,fill:true}}]}},
    options:{{responsive:true,plugins:{{legend:{{display:false}}}},
      scales:{{x:{{ticks:{{color:'#6b7280',maxTicksLimit:8}},grid:{{color:'#1c2029'}}}},
               y:{{ticks:{{color:'#6b7280'}},grid:{{color:'#1c2029'}}}}}}}}}});
}}
mk('loss','loss',D.loss,'#4ade80');
mk('lr','lr',D.lr,'#60a5fa');
mk('mfu','mfu',D.mfu,'#f59e0b',100);
mk('tps','tok/s',D.tps,'#a78bfa');
mk('gn','grad_norm',D.gn,'#f87171');
mk('val','val_loss',D.val,'#34d399');
</script></body></html>"""


def kpi(label, val, fmt="{:.4f}"):
    v = fmt.format(val) if isinstance(val, (int, float)) else str(val)
    return f'<div class="kpi"><div class="v">{v}</div><div class="l">{label}</div></div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="metrics.jsonl 경로")
    ap.add_argument("--out", default=None, help="출력 HTML (기본: 같은 폴더 dashboard.html)")
    args = ap.parse_args()

    if not os.path.exists(args.metrics):
        print(f"메트릭 파일 없음: {args.metrics}")
        sys.exit(1)
    rows = load_metrics(args.metrics)
    if not rows:
        print("메트릭이 비어 있습니다.")
        sys.exit(1)

    def s(k):
        x, y = series(rows, k)
        return {"x": x, "y": y}

    data = {
        "loss": s("train/loss"), "lr": s("train/lr"), "mfu": s("perf/mfu"),
        "tps": s("perf/tokens_per_sec"), "gn": s("train/grad_norm"),
        "val": s("val/loss"),
    }

    last = rows[-1]
    kpis = "".join([
        kpi("최신 loss", last.get("train/loss", float("nan"))),
        kpi("최신 lr", last.get("train/lr", float("nan")), "{:.2e}"),
        kpi("MFU", last.get("perf/mfu", 0) * 100, "{:.1f}%"),
        kpi("tok/s", last.get("perf/tokens_per_sec", 0) / 1e3, "{:.1f}k"),
        kpi("step", last.get("step", 0), "{:d}"),
    ])

    html = HTML.format(src=os.path.basename(args.metrics), n=len(rows),
                       last=last.get("step", 0), kpis=kpis,
                       data=json.dumps(data))
    out = args.out or os.path.join(os.path.dirname(args.metrics) or ".", "dashboard.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"대시보드 생성: {out}  ({len(rows)}개 로그)")


if __name__ == "__main__":
    main()

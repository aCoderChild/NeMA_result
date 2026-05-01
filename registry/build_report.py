#!/usr/bin/env python3
import os
import csv
import html
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/gangstat/NeMA_result")
REGISTRY_DIR = ROOT / "registry"
ANALYSIS_DIR = ROOT / "analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"

REPORT_PATH = REGISTRY_DIR / "report.html"

PIPELINE_REGISTRY = REGISTRY_DIR / "pipeline_registry.csv"
ARTIFACT_INVENTORY = REGISTRY_DIR / "artifact_inventory.csv"
METRIC_DICTIONARY = REGISTRY_DIR / "metric_dictionary.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def rel_to_registry(path: Path) -> str:
    return os.path.relpath(path, start=REGISTRY_DIR)


def html_table(rows: list[dict[str, str]], max_rows: int | None = None) -> str:
    if not rows:
        return "<p class='muted'>No data found.</p>"
    headers = list(rows[0].keys())
    body_rows = rows if max_rows is None else rows[:max_rows]
    thead = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    tbody_chunks = []
    for row in body_rows:
        tds = "".join(f"<td>{html.escape(str(row.get(h, '')))}</td>" for h in headers)
        tbody_chunks.append(f"<tr>{tds}</tr>")
    return (
        "<div class='table-wrap'><table>"
        f"<thead><tr>{thead}</tr></thead>"
        f"<tbody>{''.join(tbody_chunks)}</tbody>"
        "</table></div>"
    )


def figure_card(path: Path) -> str:
    rel = rel_to_registry(path)
    title = path.name
    return (
        "<div class='card figure-card'>"
        f"<h4>{html.escape(title)}</h4>"
        f"<a href='{html.escape(rel)}' target='_blank'><img src='{html.escape(rel)}' alt='{html.escape(title)}'></a>"
        f"<p class='path'>{html.escape(rel)}</p>"
        "</div>"
    )


def collect_figures() -> list[Path]:
    if not FIGURES_DIR.exists():
        return []
    return sorted(FIGURES_DIR.glob("*.png"))


def collect_mislang_csv_previews() -> dict[str, list[dict[str, str]]]:
    previews: dict[str, list[dict[str, str]]] = {}
    for p in sorted(ANALYSIS_DIR.glob("mislang_model_*.csv")):
        previews[p.name] = read_csv(p)
    return previews


def build_html() -> str:
    pipeline_rows = read_csv(PIPELINE_REGISTRY)
    artifact_rows = read_csv(ARTIFACT_INVENTORY)
    metric_rows = read_csv(METRIC_DICTIONARY)
    figure_paths = collect_figures()
    mislang_previews = collect_mislang_csv_previews()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    figure_grid = (
        "<div class='grid figures'>"
        + "".join(figure_card(p) for p in figure_paths)
        + "</div>"
        if figure_paths
        else "<p class='muted'>No PNG figures found in analysis/figures.</p>"
    )

    preview_blocks = []
    for name, rows in mislang_previews.items():
        preview_blocks.append(
            "<section class='subsection'>"
            f"<h4>{html.escape(name)} (first 12 rows)</h4>"
            f"{html_table(rows, max_rows=12)}"
            "</section>"
        )
    preview_html = "".join(preview_blocks) if preview_blocks else "<p class='muted'>No mislang_model CSV files found.</p>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NeMA Report</title>
  <style>
    :root {{
      --bg: #0f1115;
      --panel: #161a22;
      --panel-2: #1d2330;
      --text: #e8edf2;
      --muted: #aeb8c4;
      --border: #2b3342;
      --accent: #79b8ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    .container {{
      width: min(1400px, 95vw);
      margin: 20px auto 48px;
    }}
    h1, h2, h3, h4 {{
      margin: 0 0 8px;
      line-height: 1.2;
    }}
    h1 {{ font-size: 28px; }}
    h2 {{
      margin-top: 26px;
      font-size: 20px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 6px;
    }}
    .meta {{
      color: var(--muted);
      margin-bottom: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 14px;
    }}
    .grid {{
      display: grid;
      gap: 12px;
    }}
    .figures {{
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    }}
    .figure-card {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
    }}
    .figure-card img {{
      width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #0a0d13;
    }}
    .path {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
      word-break: break-all;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 920px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
      font-size: 13px;
      vertical-align: top;
    }}
    th {{
      background: #202635;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tr:hover td {{
      background: #202838;
    }}
    .muted {{ color: var(--muted); }}
    .quick-links a {{
      color: var(--accent);
      text-decoration: none;
      margin-right: 12px;
    }}
    .quick-links a:hover {{ text-decoration: underline; }}
    .subsection {{ margin-bottom: 16px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>NeMA Full Report</h1>
    <p class="meta">Generated at: {html.escape(generated_at)} | Root: {html.escape(str(ROOT))}</p>
    <div class="panel quick-links">
      <a href="#figures">Figures</a>
      <a href="#pipeline">Pipeline Registry</a>
      <a href="#artifacts">Artifact Inventory</a>
      <a href="#metrics">Metric Dictionary</a>
      <a href="#mislang-preview">Mislang CSV Preview</a>
    </div>

    <h2 id="figures">Figures</h2>
    <div class="panel">
      {figure_grid}
    </div>

    <h2 id="pipeline">Pipeline Registry</h2>
    <div class="panel">
      {html_table(pipeline_rows)}
    </div>

    <h2 id="artifacts">Artifact Inventory</h2>
    <div class="panel">
      {html_table(artifact_rows)}
    </div>

    <h2 id="metrics">Metric Dictionary</h2>
    <div class="panel">
      {html_table(metric_rows)}
    </div>

    <h2 id="mislang-preview">Mislang CSV Preview</h2>
    <div class="panel">
      {preview_html}
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    REPORT_PATH.write_text(build_html(), encoding="utf-8")
    print(f"Report generated: {REPORT_PATH}")


if __name__ == "__main__":
    main()

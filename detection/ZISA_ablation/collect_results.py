from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT_DIR = Path(__file__).resolve().parent
WORK_ROOT = ROOT_DIR / 'work_dirs'
OUTPUT_CSV = ROOT_DIR / 'zisa_ablation_results.csv'
EXPERIMENTS = [
    'base',
    'zisa_zoom',
    'zisa_mhsa_retry',
    'zisa_zoom_mhsa_retry',
#     'zisa_zoom_mhsa_se',
#     'zisa_full',
]

METRIC_ALIASES = {
    'mAP': ['coco/bbox_mAP', 'bbox_mAP', 'mAP'],
    'AP50': ['coco/bbox_mAP_50', 'bbox_mAP_50', 'AP50'],
    'AP75': ['coco/bbox_mAP_75', 'bbox_mAP_75', 'AP75'],
    'APs': ['coco/bbox_mAP_s', 'bbox_mAP_s', 'APs'],
    'APm': ['coco/bbox_mAP_m', 'bbox_mAP_m', 'APm'],
    'APl': ['coco/bbox_mAP_l', 'bbox_mAP_l', 'APl'],
}


def _try_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metrics(record: Dict) -> Optional[Dict[str, float]]:
    metrics: Dict[str, float] = {}
    for target_name, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if alias in record:
                parsed = _try_float(record[alias])
                if parsed is not None:
                    metrics[target_name] = parsed
                    break
    return metrics or None


def _flatten_json(obj) -> Iterable[Dict]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _flatten_json(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _flatten_json(item)


def _parse_json_file(path: Path) -> List[Dict[str, float]]:
    text = path.read_text(encoding='utf-8', errors='ignore').strip()
    if not text:
        return []

    metrics: List[Dict[str, float]] = []
    try:
        data = json.loads(text)
        for record in _flatten_json(data):
            extracted = _extract_metrics(record)
            if extracted is not None:
                metrics.append(extracted)
        return metrics
    except json.JSONDecodeError:
        pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        extracted = _extract_metrics(record)
        if extracted is not None:
            metrics.append(extracted)
    return metrics


def _find_best_metrics(exp_dir: Path) -> Optional[Dict[str, float]]:
    if not exp_dir.exists():
        return None

    candidates = sorted(
        {
            *exp_dir.rglob('*.log.json'),
            *exp_dir.rglob('scalars.json'),
            *exp_dir.rglob('*.json'),
        }
    )

    best: Optional[Dict[str, float]] = None
    best_score = -1
    for candidate in candidates:
        records = _parse_json_file(candidate)
        for record in records:
            score = len(record)
            if 'mAP' in record:
                score += 10
            if score >= best_score:
                best = record
                best_score = score
    return best


def main() -> None:
    rows = []
    for experiment in EXPERIMENTS:
        metrics = _find_best_metrics(WORK_ROOT / experiment) or {}
        rows.append(
            {
                'experiment': experiment,
                'mAP': metrics.get('mAP', ''),
                'AP50': metrics.get('AP50', ''),
                'AP75': metrics.get('AP75', ''),
                'APs': metrics.get('APs', ''),
                'APm': metrics.get('APm', ''),
                'APl': metrics.get('APl', ''),
            }
        )

    with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=['experiment', 'mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f'Results saved to: {OUTPUT_CSV}')


if __name__ == '__main__':
    main()

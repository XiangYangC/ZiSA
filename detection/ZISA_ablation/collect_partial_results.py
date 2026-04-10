from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK_ROOT = ROOT / 'work_dirs'

EXPERIMENTS = [
    'base',
    'zisa_zoom',
    'zisa_mhsa_retry',
    'zisa_zoom_mhsa_retry',
    'zisa_zoom_mhsa_se_retry',
    'zisa_full_retry',
]

METRIC_KEYS = [
    'coco/bbox_mAP',
    'coco/bbox_mAP_50',
    'coco/bbox_mAP_75',
    'coco/bbox_mAP_s',
    'coco/bbox_mAP_m',
    'coco/bbox_mAP_l',
]


def find_latest_scalars(exp_dir: Path) -> Path | None:
    candidates = sorted(exp_dir.rglob('vis_data/scalars.json'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_last_metrics(path: Path) -> dict[str, float] | None:
    last_record = None
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'coco/bbox_mAP' in record:
                last_record = record
    if last_record is None:
        return None
    return {key: last_record.get(key, '') for key in METRIC_KEYS}


def main() -> None:
    print('experiment,mAP,AP50,AP75,APs,APm,APl,source')
    for exp in EXPERIMENTS:
        exp_dir = WORK_ROOT / exp
        scalars = find_latest_scalars(exp_dir)
        if scalars is None:
            print(f'{exp},,,,,,,NOT_FOUND')
            continue

        metrics = read_last_metrics(scalars)
        if metrics is None:
            print(f'{exp},,,,,,,{scalars}')
            continue

        print(
            f"{exp},"
            f"{metrics['coco/bbox_mAP']},"
            f"{metrics['coco/bbox_mAP_50']},"
            f"{metrics['coco/bbox_mAP_75']},"
            f"{metrics['coco/bbox_mAP_s']},"
            f"{metrics['coco/bbox_mAP_m']},"
            f"{metrics['coco/bbox_mAP_l']},"
            f"{scalars}"
        )


if __name__ == '__main__':
    main()

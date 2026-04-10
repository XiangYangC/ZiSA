# ZiSA

ZiSA is a customized object detection project built on top of `VMamba` and `MMDetection`, focused on windfarm detection experiments. This repository contains the backbone adaptation, attention modules, ablation code, visualization scripts, and training configs used for the ZiSA experiments.

## Overview

Compared with the original VMamba-based detector, this project adds a ZiSA enhancement module for feature interaction and target-focused refinement in the detection pipeline.

Main components in this repository:

- `detection/attention_modules.py`: ZiSA related attention modules
- `detection/enhanced_fpn.py`: enhanced FPN implementation
- `detection/configs/windfarm/`: windfarm detection configs
- `detection/ZISA_ablation/`: ablation study code
- `detection/ZiSAvisualization/`: feature and heatmap visualization tools

## Results

### VMamba-based comparison

| Model / Method | Backbone | MHSA | Zoom | SE | Gate | mAP | mAP50 | mAP75 | mAP_s | mAP_m | mAP_l |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VMamba (Base) | VMamba | X | X | X | X | 0.508 | 0.813 | 0.540 | 0.601 | 0.476 | 0.586 |
| VMamba + ZiSA | VMamba | check | check | check | check | **0.516** | **0.821** | **0.551** | **0.623** | **0.482** | **0.613** |

### ResNet50 ablation

| Model / Method | Backbone | MHSA | Zoom | SE | Gate | mAP | mAP50 | mAP75 | mAP_s | mAP_m | mAP_l |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | ResNet50 | X | X | X | X | 0.820 | 0.841 | 0.820 | 0.931 | 0.801 | 0.846 |
| +MHSA | ResNet50 | check | X | X | X | 0.819 | 0.841 | 0.827 | 0.915 | 0.804 | 0.850 |
| +Zoom | ResNet50 | X | check | X | X | 0.823 | 0.843 | 0.819 | 0.918 | 0.808 | 0.851 |
| +MHSA + Zoom | ResNet50 | check | check | X | X | **0.825** | **0.848** | **0.832** | 0.921 | 0.807 | 0.851 |
| +MHSA + Zoom + SE | ResNet50 | check | check | check | X | 0.823 | 0.841 | 0.828 | 0.930 | 0.802 | 0.851 |
| Full (ZiSA) | ResNet50 | check | check | check | check | 0.821 | 0.839 | 0.820 | 0.916 | **0.809** | 0.848 |

### Reference

| Model | mAP |
| --- | --- |
| YOLOv8 | 0.537 |

## Repository Structure

```text
.
|- vmamba.py
|- detection/
|  |- attention_modules.py
|  |- enhanced_fpn.py
|  |- model.py
|  |- train_windfarm_detection.py
|  |- configs/
|  |  |- windfarm/
|  |  |  |- faster_rcnn_r50_windfarm.py
|  |  |  |- faster_rcnn_r18_zisa.py
|  |  |  |- faster_rcnn_vssm_windfarm_with_attention.py
|  |  |  |- mask_rcnn_vssm_windfarm.py
|  |- ZISA_ablation/
|  |- ZiSAvisualization/
|- classification/
|- segmentation/
|- kernels/
```

## Environment Setup

Recommended environment:

```bash
conda create -n zisa python=3.10
conda activate zisa
pip install -r requirements.txt
cd kernels/selective_scan
pip install .
cd ../..
```

If you need detection dependencies:

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```

## Training

### Windfarm detection

Use the custom training script:

```bash
python detection/train_windfarm_detection.py
```

If you want to switch configs, edit the config path inside the training script or directly run MMDetection:

```bash
python detection/tools/train.py detection/configs/windfarm/faster_rcnn_r18_zisa.py
```

Other useful configs:

- `detection/configs/windfarm/faster_rcnn_r50_windfarm.py`
- `detection/configs/windfarm/faster_rcnn_r18_zisa.py`
- `detection/configs/windfarm/faster_rcnn_vssm_windfarm_with_attention.py`
- `detection/configs/windfarm/mask_rcnn_vssm_windfarm.py`

## Ablation

ZiSA ablation code is under:

- `detection/ZISA_ablation/run_ablation.sh`
- `detection/ZISA_ablation/configs/base.py`
- `detection/ZISA_ablation/configs/zisa_mhsa.py`
- `detection/ZISA_ablation/configs/zisa_zoom.py`
- `detection/ZISA_ablation/configs/zisa_zoom_mhsa.py`
- `detection/ZISA_ablation/configs/zisa_zoom_mhsa_se.py`
- `detection/ZISA_ablation/configs/zisa_full.py`

## Visualization

Visualization scripts are under `detection/ZiSAvisualization/`.

Examples:

```bash
python detection/ZiSAvisualization/run_zisa_visualization.py
python detection/ZiSAvisualization/run_detection_visualization.py
```

## Base Projects

This repository is based on:

- `VMamba`
- `MMDetection`
- `OpenMMLab`

and extends them for ZiSA-based windfarm object detection experiments.

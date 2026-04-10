bash ./detection/tools/dist_test.sh \
    detection/configs/windfarm/faster_rcnn_vssm_windfarm_with_attention.py \
    work_dirs/windfarm_detection_with_attention/epoch_12.pth \
    1 \
    --format-only \
    --eval-options "jsonfile_prefix=./work_dirs/windfarm_detection_with_attention/results"
tools/cg_restore_train_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_kp/solver.prototxt \
	--weights output/kp/aichal_train/zf_faster_rcnn_iter_90000.solverstate \
    --iters 100000 \
    --cfg cfgs/kp.yml \
    --imdb aichal2_2017_train &> log_aichal_restore


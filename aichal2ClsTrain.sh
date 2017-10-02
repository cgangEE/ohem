tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_kpCls/solver.prototxt \
	--weights output/kp/aichal_train/zf_faster_rcnn_iter_100000.caffemodel \
    --iters 130000 \
    --cfg cfgs/kpCls.yml \
    --imdb aichal2_2017_train #&> log_aichal


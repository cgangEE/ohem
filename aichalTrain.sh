tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_kp/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/kp.yml \
    --imdb aichal_2017_train &> log_aichal


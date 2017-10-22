tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/fcn/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/fcn.yml \
    --imdb aichalCrop_2017_train &> log_fcn_aichalCrop


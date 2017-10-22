tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/fcn_reg/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/fcnReg.yml \
    --imdb aichalCrop_2017_train &> log_fcn_reg_aichalCrop


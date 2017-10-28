tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/fcnV1_reg/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/fcnV1Reg.yml \
    --imdb aichalCrop_2017_train &> log_fcnV1_reg_aichalCrop


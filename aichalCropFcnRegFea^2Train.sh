tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/fcn_reg_fea^2/solver.prototxt \
	--weights models/fcn_reg_fea^2/original_fea^2.model \
    --iters 100000 \
    --cfg cfgs/fcnRegFea^2.yml \
    --imdb aichalCrop_2017_train &> log_fcn_reg_fea^2_aichalCrop


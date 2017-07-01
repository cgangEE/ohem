tools/cg_train_head_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_UpperBody_Ohem/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFull_DRoiAlignX_UpperBody_Ohem.yml \
    --imdb psdbUpperBody_2015_train &> log_psdbUpperBody_DRoiAlignX_Ohem



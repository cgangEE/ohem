tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_DRoiAlignX.yml \
    --imdb ua_2016_train &> log_ua_DRoiAlignX


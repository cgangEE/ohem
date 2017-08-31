tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_DRoiAlignX.yml \
    --imdb psdbCrop_2015_train &> log_psdbCrop


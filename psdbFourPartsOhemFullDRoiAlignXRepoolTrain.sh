# with ohem
tools/cg_train_head_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_FourParts_Ohem_Repool/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFull_DRoiAlignX_FourParts_Ohem_Repool.yml \
    --imdb psdbFourParts_2015_train &> log_psdbFourParts_DRoiAlignX_Ohem_Repool



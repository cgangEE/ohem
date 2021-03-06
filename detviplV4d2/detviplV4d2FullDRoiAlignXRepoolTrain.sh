tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_Repool/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_DRoiAlignX_Repool.yml \
    --imdb detviplV4d2_2016_train &> log_detviplV4d2_DRoiAlignX_Repool


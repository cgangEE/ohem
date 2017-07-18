tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/full4_DRoiAlignX/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_DRoiAlignX.yml \
    --imdb detviplV7d1_2016_train &>> log_detviplV7d1_DRoiAlignX



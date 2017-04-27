tools/train_net.py  \
    --gpu 0 \
    --solver models/full4_DRoiAlignX/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_DRoiAlignX.yml \
    --imdb psdb_2015_train &> log_psdb_DRoiAlignX



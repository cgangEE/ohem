tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/full3_DRoiAlignX/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_DRoiAlignX.yml \
    --imdb part_2016_train &> log_part_DRoiAlignX


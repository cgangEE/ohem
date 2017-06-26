tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_Fast/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_DRoiAlignX_Fast.yml \
    --imdb detviplV4d2_2016_train &> log_detviplV4d2_DRoiAlignX_Fast


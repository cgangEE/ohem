tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_Face/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_DRoiAlignX_Face.yml \
    --imdb wider_2015_train &> log_wider_DRoiAlignX_Face


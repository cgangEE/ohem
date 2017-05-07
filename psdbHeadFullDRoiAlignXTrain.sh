# without ohem
tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_Head/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_DRoiAlignX_Head.yml \
    --imdb psdbHead_2015_train &> log_psdb_DRoiAlignX_Head



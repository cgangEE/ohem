# with ohem
tools/cg_train_head_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_Head_Ohem/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFull_DRoiAlignX_Head_Ohem.yml \
    --imdb psdbHead_2015_train &> log_psdb_DRoiAlignX_Head_Ohem



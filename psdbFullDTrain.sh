tools/train_net.py  \
    --gpu 0 \
    --solver models/full4_D/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_D.yml \
    --imdb psdb_2015_train &> log_psdb_D 



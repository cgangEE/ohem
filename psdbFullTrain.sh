tools/train_net.py  \
    --gpu 0 \
    --solver models/full4/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem.yml \
    --imdb psdb_2015_train &> log_psdb



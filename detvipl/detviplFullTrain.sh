tools/train_net.py  \
    --gpu 0 \
    --solver models/pvanet/full1_ohem/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg models/pvanet/cfgs/trainFullOhem.yml \
    --imdb detvipl_2016_train &> log_detvipl &


tools/train_net.py  \
    --gpu 3 \
    --solver models/pvanet/full7_ohem/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg models/pvanet/cfgs/trainFull7Ohem.yml \
    --imdb fish_2016_train &> log_fishFull &


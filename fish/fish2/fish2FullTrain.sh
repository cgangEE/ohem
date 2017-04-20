tools/train_net.py  \
    --gpu 0 \
    --solver models/pvanet/full1/solver.prototxt \
	--weights models/pvanet/full/original.model \
    --iters 150000 \
    --cfg models/pvanet/cfgs/trainFullFish.yml \
    --imdb fish2_2016_train &> log_fish2Full &


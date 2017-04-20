tools/train_net.py  \
    --gpu 0 \
    --solver models/pvanet/full1_ohem/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem.yml \
    --imdb detviplV4d2_2016_train &> log_detviplV4d2&


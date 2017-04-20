tools/train_net.py  \
    --gpu 0 \
    --solver models/pvanet/full1_ohem/solver.prototxt \
	--weights psdbVeh_pvanet_100000.caffemodel \
    --iters 100000 \
    --cfg models/pvanet/cfgs/trainFullOhem.yml \
    --imdb sjtu_ia_2016_train &> log_sjtuFullOhem &


tools/train_net.py  \
    --gpu 0 \
    --solver models/full20_ohem_D/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_D.yml \
    --imdb voc_2007_trainval &> log_voc_D &


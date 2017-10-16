tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/wider_cluster/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/wider_cluster.yml \
    --imdb wider_2015_train &> log_wider_cluster


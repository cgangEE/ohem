tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/kp_cluster/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/kpCluster.yml \
    --imdb aichal2_2017_train &> log_kpCluster_aichal2


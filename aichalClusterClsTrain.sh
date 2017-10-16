tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/kp_cluster_cls/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/kpClusterCls.yml \
    --imdb aichal_2017_train &> log_aichal_cluster_cls


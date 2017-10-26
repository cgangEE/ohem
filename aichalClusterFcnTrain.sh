tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/kp_cluster_fcn/solver.prototxt \
	--weights  models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/kpClusterFcn.yml \
    --imdb aichal_2017_train &> log_kpClusterFcn_aichal


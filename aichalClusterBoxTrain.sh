tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/aichal_cluster/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/aichalCluster.yml \
    --imdb aichal_2017_train &> log_bboxCluster_aichal


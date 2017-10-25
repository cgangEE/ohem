tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/kp_cluster_fcn/solver.prototxt \
	--weights output/aichalCluster/aichal_train/zf_faster_rcnn_iter_100000.caffemodel \
    --iters 100000 \
    --cfg cfgs/kpClusterFcn.yml \
    --imdb aichal_2017_train &> log_kpClusterFcn_aichal

# models/pvanet/original.model \

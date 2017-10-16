tools/cg_restore_train_net.py  \
    --gpu 0 \
    --solver models/wider_cluster/solver.prototxt \
	--weights output/wider_cluster/wider_train/zf_faster_rcnn_iter_60000.solverstate \
    --iters 100000 \
    --cfg cfgs/wider_cluster.yml \
    --imdb wider_2015_train &> log_wider_cluster_restore


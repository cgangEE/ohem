tools/train_net.py  \
    --gpu 0 \
    --solver models/full1_D/solver.prototxt \
	--weights output/pvanet_full1_ohem_D/psdb_train/zf_faster_rcnn_iter_100000.caffemodel \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_DPSDB.yml \
    --imdb detviplV4d2_2016_train &> log_detviplV4d2_DPSDB 



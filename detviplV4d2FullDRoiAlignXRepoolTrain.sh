tools/cg_train_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_Repool/solver.prototxt \
	--weights  output/pvanet_full1_ohem_DRoiAlignX/detviplV4d2_train/zf_faster_rcnn_iter_100000.caffemodel \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_DRoiAlignX_Repool.yml \
    --imdb detviplV4d2_2016_train &> log_detviplV4d2_DRoiAlignX_Repool


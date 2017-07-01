tools/cg_restore_train_net.py  \
    --gpu 0 \
    --solver models/full4_DRoiAlignX/solver.prototxt \
	--weights output/pvanet_full1_ohem_DRoiAlignX/detviplV7d0_train/zf_faster_rcnn_iter_40000.solverstate \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_DRoiAlignX.yml \
    --imdb detviplV7d0_2016_train &>> log_detviplV7d0_DRoiAlignX



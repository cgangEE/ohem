# with ohem
tools/cg_train_head_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_FourParts_Ohem_Repool_Loss1/solver.prototxt \
	--weights output/pvanet_full1_DRoiAlignX_FourParts_Ohem/psdbFourParts_train/zf_faster_rcnn_iter_100000.caffemodel \
    --iters 100000 \
    --cfg cfgs/trainFull_DRoiAlignX_FourParts_Ohem_Repool_Loss1.yml \
    --imdb psdbFourParts_2015_train &> log_psdbFourParts_DRoiAlignX_Ohem_Repool_Loss1



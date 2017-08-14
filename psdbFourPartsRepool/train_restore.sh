# with ohem
tools/cg_restore_train_head_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_FourParts_Ohem_Repool/solver.prototxt \
	--weights output/pvanet_full1_DRoiAlignX_FourParts_Ohem_Repool/psdbFourParts_train/zf_faster_rcnn_iter_50000.solverstate \
    --iters 100000 \
    --cfg cfgs/trainFull_DRoiAlignX_FourParts_Ohem_Repool.yml \
    --imdb psdbFourParts_2015_train &>> log_psdbFourParts_DRoiAlignX_Ohem_Repool



# with ohem
tools/cg_restore_train_head_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_FourParts_Ohem_RepoolRemove/solver.prototxt \
	--weights output/pvanet_full1_DRoiAlignX_FourParts_Ohem_RepoolRemove/psdbFourParts_train/zf_faster_rcnn_iter_90000.solverstate \
    --iters 100000 \
    --cfg cfgs/trainFull_DRoiAlignX_FourParts_Ohem_RepoolRemove.yml \
    --imdb psdbFourParts_2015_train &>> log_psdbFourParts_DRoiAlignX_Ohem_RepoolRemove



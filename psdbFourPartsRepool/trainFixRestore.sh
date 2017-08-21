# with ohem
tools/cg_restore_train_head_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_FourParts_Ohem_RepoolFix/solver.prototxt \
	--weights output/pvanet_full1_DRoiAlignX_FourParts_Ohem_RepoolFix/psdbFourParts_train/zf_faster_rcnn_iter_20000.solverstate \
    --iters 100000 \
    --cfg cfgs/trainFull_DRoiAlignX_FourParts_Ohem_RepoolFix.yml \
    --imdb psdbFourParts_2015_train &>> log_psdbFourParts_DRoiAlignX_Ohem_RepoolFix



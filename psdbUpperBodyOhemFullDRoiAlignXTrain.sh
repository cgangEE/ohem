tools/cg_train_head_net.py  \
    --gpu 0 \
    --solver models/full1_DRoiAlignX_UpperBody_Ohem/solver.prototxt \
	--weights output/pvanet_full1_DRoiAlignX_UpperBody/psdbUpperBody_train/zf_faster_rcnn_iter_80000.solverstate \
    --iters 100000 \
    --cfg cfgs/trainFull_DRoiAlignX_UpperBody.yml \
    --imdb psdbUpperBody_2015_train &>> log_psdbUpperBody_DRoiAlignX



tools/train_net.py  \
    --gpu 0 \
    --solver models/full1_D2/solver.prototxt \
	--weights output/pvanet_full1_ohem_D2/detviplV4d2_train/zf_faster_rcnn_iter_60000.caffemodel \
    --iters 40000 \
    --cfg cfgs/trainFullOhem_D2.yml \
    --imdb detviplV4d2_2016_train &> log_detviplV4d2_D2


tools/test_net.py \
	    --gpu 0 \
	    --def models/full1_D2/test.pt \
		--net output/pvanet_full1_ohem_D2/detviplV4d2_train/zf_faster_rcnn_iter_40000.caffemodel \
	    --cfg models/pvanet/cfgs/submit_160715_full_ohem_D2.yml \
		--imdb detviplV4d2_2016_test &> log_detviplV4d2_D2_test 

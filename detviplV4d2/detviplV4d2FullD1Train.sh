tools/train_net.py  \
    --gpu 0 \
    --solver models/full1_D1/solver.prototxt \
	--weights models/pvanet/original.model \
    --iters 100000 \
    --cfg cfgs/trainFullOhem_D1.yml \
    --imdb detviplV4d2_2016_train &> log_detviplV4d2_D1


tools/test_net.py \
	    --gpu 0 \
	    --def models/full1_D1/test.pt \
		--net output/pvanet_full1_ohem_D1/detviplV4d2_train/zf_faster_rcnn_iter_100000.caffemodel \
	    --cfg models/pvanet/cfgs/submit_160715_full_ohem_D1.yml \
		--imdb detviplV4d2_2016_test &> log_detviplV4d2_D1_test 

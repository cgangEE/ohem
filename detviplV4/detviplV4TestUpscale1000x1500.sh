tools/test_net.py \
	    --gpu 0 \
	    --def models/pvanet/full1_ohem/test_inference.prototxt \
		--net detvipl_v4_100000_inference.caffemodel \
	    --cfg models/pvanet/cfgs/submit_160715_full_upscale1000x1500.yml \
		--imdb detviplV4_2016_test  &> log_detviplv4_upscale1000x1500 &


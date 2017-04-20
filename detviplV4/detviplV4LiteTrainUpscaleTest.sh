tools/test_net.py \
	    --gpu 0 \
		--def models/pvanet/lite1_ohem/test_inference.prototxt \
		--net detvipl_v4_upscale_lite_100000_inference.caffemodel \
	    --cfg models/pvanet/cfgs/submit_160715_lite.yml \
		--imdb detviplV4_2016_test  &> log_detviplv4_lite_upscale &

:<<mark
tools/test_net.py \
	    --gpu 0 \
		--def models/pvanet/lite1_ohem/test_inference.prototxt \
		--net detvipl_v4_upscale_lite_100000_inference.caffemodel
	    --cfg models/pvanet/cfgs/submit_160715_lite_1000x1500.yml \
		--imdb detviplV4_2016_test  &> log_detviplv4_lite_upscale_1000x1500 &

tools/test_net.py \
	    --gpu 0 \
		--def models/pvanet/lite1_ohem/test_inference.prototxt \
		--net detvipl_v4_upscale_lite_100000_inference.caffemodel
	    --cfg models/pvanet/cfgs/submit_160715_lite_1500x2000.yml \
		--imdb detviplV4_2016_test  &> log_detviplv4_lite_upscale_1500x2000 &


mark

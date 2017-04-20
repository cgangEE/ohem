
:<<mark
tools/test_net.py \
	    --gpu 0 \
	    --def models/pvanet/full1_ohem/test_inference_comp.prototxt \
		--net detvipl_v4_upscale_100000_inference_svd_fc6_512_fc7_512.caffemodel \
	    --cfg models/pvanet/cfgs/submit_160715_full_svd.yml \
		--imdb detviplV4_2016_test  &> log_detviplv4_trainUpscale_svd &

tools/test_net.py \
	    --gpu 0 \
	    --def models/pvanet/full1_ohem/test_inference_comp.prototxt \
		--net detvipl_v4_upscale_100000_inference_svd_fc6_512_fc7_512.caffemodel \
	    --cfg models/pvanet/cfgs/submit_160715_full_svd_1000x1500.yml \
		--imdb detviplV4_2016_test  &> log_detviplv4_trainUpscale_svd_1000x1500 &

mark

tools/test_net.py \
	    --gpu 0 \
	    --def models/pvanet/full1_ohem/test_inference_comp.prototxt \
		--net detvipl_v4_upscale_100000_inference_svd_fc6_512_fc7_512.caffemodel \
	    --cfg models/pvanet/cfgs/submit_160715_full_svd_1500x2000.yml \
		--imdb detviplV4_2016_test  &> log_detviplv4_trainUpscale_svd_1500x2000 &

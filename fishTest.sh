tools/test_net.py \
	--gpu 0 \
	--def models/pvanet/full7_ohem/test.pt \
	--net output/pvanet_full7_ohem/fish_train/zf_faster_rcnn_iter_100000.caffemodel \
	--cfg models/pvanet/cfgs/submit_160715_full7_ohem.yml \
	--imdb fish_2016_test  &> log_fish_test &


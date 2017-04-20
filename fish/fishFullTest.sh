suffix='0000.caffemodel'
for i in {10..10}
do
	echo '>>>>>>>>>>>>Testing '$i
	tools/test_net.py \
	    --gpu 0 \
	    --def models/pvanet/full7_ohem/test.pt \
		--net output/pvanet_full7_ohem/fish_train/zf_faster_rcnn_iter_$i$suffix \
	    --cfg models/pvanet/cfgs/submit_160715_full7_ohem.yml \
		--imdb fish_2016_test
done


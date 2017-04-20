suffix='0000.caffemodel'
for i in {15..15}
do
	echo '>>>>>>>>>>>>Testing '$i
	tools/test_net.py \
	    --gpu 0 \
	    --def models/pvanet/full1/test.pt \
		--net output/pvanet_full7/fish2_train/zf_faster_rcnn_iter_$i$suffix \
	    --cfg models/pvanet/cfgs/submit_160715_full1.yml \
		--imdb fish2_2016_val
done


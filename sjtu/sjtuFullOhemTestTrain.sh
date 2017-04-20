suffix='0000.caffemodel'
for i in {1..10}
do
	echo '>>>>>>>>>>>>Testing '$i
	tools/test_net.py \
	    --gpu 0 \
	    --def models/pvanet/full1_ohem/test.pt \
		--net output/pvanet_full1_ohem/sjtu_ia_train/zf_faster_rcnn_iter_$i$suffix \
	    --cfg models/pvanet/cfgs/submit_160715_full_ohem.yml \
		--imdb sjtu_ia_2016_train
done


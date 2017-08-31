suffix='0000_inference.caffemodel'
echo 'Begin' &> log_detviplV4d2_D2_test
for i in {4..4}
do
	echo 'Testing'  $i &>> log_detviplV4d2_D2_test
	tools/test_net.py \
	    --gpu 0 \
	    --def models/full1_D2/test_inference.prototxt \
		--net output/pvanet_full1_ohem_D2/detviplV4d2_train/zf_faster_rcnn_iter_$i$suffix \
	    --cfg cfgs/submit_160715_full_ohem_D2.yml \
		--imdb detviplV4d2_2016_test &>> log_detviplV4d2_D2_test 
done

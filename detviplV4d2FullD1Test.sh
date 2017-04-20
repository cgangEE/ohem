suffix='0000.caffemodel'
echo 'Begin' &> log_detviplV4d2_D1_test
for i in {10..10}
do
	echo 'Testing'  $i &>> log_detviplV4d2_D1_test

	tools/test_net.py \
	    --gpu 0 \
	    --def models/full1_D1/test.pt \
		--net output/pvanet_full1_ohem_D1/detviplV4d2_train/zf_faster_rcnn_iter_$i$suffix \
	    --cfg cfgs/submit_160715_full_ohem_D1.yml \
		--imdb detviplV4d2_2016_test &>> log_detviplV4d2_D1_test
done

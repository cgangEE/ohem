echo 'Begin' &> log_detviplV4d2_DRoiAlign_test
for i in {9..9}
do
	echo 'Testing'  $i &>> log_detviplV4d2_DRoiAlign_test
	tools/test_net.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlign/test.pt \
		--net output/pvanet_full1_ohem_D/detviplV4d2_train/zf_faster_rcnn_iter_${i}0000.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_DRoiAlign.yml \
		--imdb detviplV4d2_2016_test &>> log_detviplV4d2_DRoiAlign_test 
done

#echo 'Begin' &> log_psdb_DRoiAlignX_test

for i in {10..10}
do

	echo 'Testing'  $i &>> log_detviplV4d2_DRoiAlignX_test_${i}

	tools/test_net.py \
	    --gpu $(($i%1)) \
	    --def models/full1_DRoiAlignX/test_inference.prototxt \
		--net output/pvanet_full1_ohem_DRoiAlignX/detviplV4d2_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_DRoiAlignX.yml \
		--imdb detviplV4d2_2016_test  &>> log_detviplV4d2_DRoiAlignX_test_${i} &
done


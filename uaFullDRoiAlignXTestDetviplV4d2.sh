for i in {10..10}
do

	echo 'Testing'  $i &>> log_ua_DRoiAlignX_test_detviplV4d2_test_${i}

	tools/test_net.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX/test_inference.prototxt \
		--net output/pvanet_full1_ohem_DRoiAlignX/ua_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_DRoiAlignX_testCar.yml \
		--imdb detviplV4d2_2016_test &>> log_ua_DRoiAlignX_test_detviplV4d2_test_${i} &
done


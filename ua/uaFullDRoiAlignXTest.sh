#echo 'Begin' &> log_psdb_DRoiAlignX_test

for i in {9..10}
do

	echo 'Testing'  $i &>> log_ua_DRoiAlignX_test_${i}

	tools/test_net.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX/test_inference.prototxt \
		--net output/pvanet_full1_ohem_DRoiAlignX/ua_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_DRoiAlignX.yml \
		--imdb ua_2016_test  &>> log_ua_DRoiAlignX_test_${i} &
done


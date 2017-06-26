#echo 'Begin' &> log_psdbUpperBody_DRoiAlignX_test

for i in {10..1}
do

	echo 'Testing'  $i &>> log_psdbUpperBody_DRoiAlignX_test_${i}

	tools/test_net_upper_body.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX_UpperBody/test_inference.prototxt \
		--net output/pvanet_full1_DRoiAlignX_UpperBody/psdbUpperBody_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_DRoiAlignX_UpperBody.yml \
		--imdb psdbUpperBody_2015_test  &>> log_psdbUpperBody_DRoiAlignX_test_${i} 
done


#echo 'Begin' &> log_psdbUpperBody_DRoiAlignX_test

for i in {10..10}
do

	echo 'Testing'  $i &>> log_psdbUpperBody_Ohem-C_DRoiAlignX_test_${i}

	tools/test_net_upper_body.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX_UpperBody_Ohem-C/test_inference.prototxt \
		--net output/pvanet_full1_DRoiAlignX_UpperBody_Ohem-C/psdbUpperBody_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_DRoiAlignX_UpperBody_Ohem-C.yml \
		--imdb psdbUpperBody_2015_test  &>> log_psdbUpperBody_Ohem-C_DRoiAlignX_test_${i} 
done


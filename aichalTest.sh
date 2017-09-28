for i in {10..10}
do
	echo 'Testing'  $i &>> log_aichal_test_${i}
	tools/test_net_kp.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX_kp/test_inference.prototxt \
		--net output/kp/aichal_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/kp.yml \
		--imdb aichal_2017_test &>> log_aichal_test_${i} 
done


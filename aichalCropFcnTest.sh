for i in {10..10}
do
	echo 'Testing'  $i &>> log_fcn_aichalCrop_test_${i}
	tools/test_net_fcn.py \
	    --gpu 0 \
	    --def models/fcn/test_inference.prototxt \
		--net output/fcn/aichalCrop_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/fcn.yml \
		--imdb aichalCrop_2017_test &>> log_fcn_aichalCrop_test_${i} 
done


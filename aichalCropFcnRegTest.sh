for i in {10..10}
do
	echo 'Testing'  $i &>> log_fcn_reg_aichalCrop_test_${i}
	tools/test_net_fcn_reg.py \
	    --gpu 0 \
	    --def models/fcn_reg/test_inference.prototxt \
		--net output/fcnReg/aichalCrop_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/fcnReg.yml \
		--imdb aichalCrop_2017_test &>> log_fcn_reg_aichalCrop_test_${i} 
done


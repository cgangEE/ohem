for i in {10..10}
do
	echo 'Testing'  $i &>> log_fcnV1_reg_aichalCrop_val_${i}
	tools/test_net_fcn_reg.py \
	    --gpu 0 \
	    --def models/fcnV1_reg/test_inference.prototxt \
		--net output/fcnV1Reg/aichalCrop_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/fcnV1Reg.yml \
		--imdb aichalCrop_2017_val &>> log_fcnV1_reg_aichalCrop_val_${i} 
done


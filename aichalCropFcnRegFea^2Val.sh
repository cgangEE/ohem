for i in {10..10}
do
	echo 'Testing'  $i &>> log_fcn_reg_fea^2_aichalCrop_val_${i}
	tools/test_net_fcn_reg^2.py \
	    --gpu 0 \
	    --def models/fcn_reg_fea^2/test_inference.prototxt \
		--net output/fcnRegFea^2/aichalCrop_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/fcnRegFea^2.yml \
		--imdb aichalCrop_2017_val &>> log_fcn_reg_fea^2_aichalCrop_val_${i} 
done


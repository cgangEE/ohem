for i in {10..10}
do
	echo 'Testing'  $i &>> log_psdbCrop_DRoiAlignX_val2_${i}
	tools/test_net.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX/test_inference.prototxt \
		--net output/pvanet_full1_ohem_DRoiAlignX/psdbCrop_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_DRoiAlignX.yml \
		--imdb psdbCrop_2015_val2  &>> log_psdbCrop_DRoiAlignX_val2_${i} 
done


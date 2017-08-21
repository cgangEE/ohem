for i in {10..1}
do

	echo 'Testing'  $i &>> log_psdbFourParts_Ohem_DRoiAlignX_RepoolFix_test_${i}

	tools/test_net_four_parts_repoolFix.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX_FourParts_Ohem_RepoolFix/test_inference.prototxt \
		--net output/pvanet_full1_DRoiAlignX_FourParts_Ohem_RepoolFix/psdbFourParts_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_DRoiAlignX_FourParts_Ohem_RepoolFix.yml \
		--imdb psdbFourParts_2015_test  &>> log_psdbFourParts_Ohem_DRoiAlignX_RepoolFix_test_${i} 
done


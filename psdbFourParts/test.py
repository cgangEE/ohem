for i in {10..10}
do
	echo 'Testing'  $i &>> log_psdbFourParts_Ohem_DRoiAlignX_test_${i}
	tools/test_net_four_parts.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX_FourParts_Ohem/test_inference.prototxt \
		--net output/pvanet_full1_DRoiAlignX_FourParts_Ohem/psdbFourParts_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_DRoiAlignX_FourParts_Ohem.yml \
		--imdb psdbFourParts_2015_test  &>> log_psdbFourParts_Ohem_DRoiAlignX_test_${i} 
done


for i in {3..3}
do

	echo 'Testing'  $i &>> log_psdbFourParts_Ohem_DRoiAlignX_Repool_test_${i}

	tools/test_net_four_parts_repool.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX_FourParts_Ohem_Repool/test.pt \
		--net output/pvanet_full1_DRoiAlignX_FourParts_Ohem/psdbFourParts_train/zf_faster_rcnn_iter_100000.caffemodel \
	    --cfg cfgs/submit_160715_full_DRoiAlignX_FourParts_Ohem_Repool.yml \
		--imdb psdbFourParts_2015_test  &>> log_psdbFourParts_Ohem_DRoiAlignX_Repool_test_${i} 
done


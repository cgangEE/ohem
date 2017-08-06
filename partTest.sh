for i in {10..10}
do

	echo 'Testing'  $i &>> log_part_DRoiAlignX_test_${i}

	tools/test_net.py \
	    --gpu $(($i%1)) \
	    --def models/full3_DRoiAlignX/test_inference.prototxt \
		--net output/pvanet_full1_ohem_DRoiAlignX/part_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_DRoiAlignX.yml \
		--imdb part_2016_test  &> log_part_DRoiAlignX_test_${i} 
done


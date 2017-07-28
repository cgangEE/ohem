for i in {9..1}
do

	echo 'Testing'  $i &>> log_detviplV7d2d1_DRoiAlignX_test_${i}

	tools/test_net.py \
	    --gpu $(($i%1)) \
	    --def models/full4_DRoiAlignX/test_inference.prototxt \
		--net output/pvanet_full1_ohem_DRoiAlignX/detviplV7d2_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_DRoiAlignX.yml \
		--imdb detviplV7d2d1_2016_test  &> log_detviplV7d2d1_DRoiAlignX_test_${i} 
done


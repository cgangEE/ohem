for i in {10..10}
do
	echo 'Testing'  $i &> log_wider_DRoiAlignX_test_fddb_300_${i}
	tools/test_net.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX/test_inference.prototxt \
		--net output/pvanet_full1_ohem_DRoiAlignX/wider_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_DRoiAlignX_300.yml \
		--imdb fddb_2015_test  &> log_wider_DRoiAlignX_test_fddb_300_${i} 
done


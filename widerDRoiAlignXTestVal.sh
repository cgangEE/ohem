for i in {10..10}
do
	echo 'Testing'  $i &>> log_wider_DRoiAlignX_test_val_${i}
	tools/test_net.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX/test_inference.prototxt \
		--net output/pvanet_full1_ohem_DRoiAlignX/wider_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_DRoiAlignX.yml \
		--imdb wider_2015_val &>> log_wider_DRoiAlignX_test_val_${i} 
done


#echo 'Begin' &> log_psdb_DRoiAlignX_test

for i in {10..10}
do

	echo 'Testing'  $i &>> log_psdbCrop_DRoiAlignX_val1_${i}

	tools/test_net.py \
	    --gpu 1 \
	    --def models/full1_DRoiAlignX/test_inference.prototxt \
		--net output/pvanet_full1_ohem_DRoiAlignX/psdbCrop_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_DRoiAlignX.yml \
		--imdb psdbCrop_2015_val1  &>> log_psdbCrop_DRoiAlignX_val1_${i} 
done


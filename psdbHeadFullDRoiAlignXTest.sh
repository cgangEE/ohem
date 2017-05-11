#echo 'Begin' &> log_psdbHead_DRoiAlignX_test

for i in {10..10}
do

	echo 'Testing'  $i &>> log_psdbHead_DRoiAlignX_test_${i}

	tools/test_net_head.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX_Head/test_inference.prototxt \
		--net output/pvanet_full1_DRoiAlignX_Head/psdbHead_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_DRoiAlignX_Head.yml \
		--imdb psdbHead_2015_test  &>> log_psdbHead_DRoiAlignX_test_${i} &
done


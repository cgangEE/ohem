echo 'Begin' &> log_riding_DRoiAlignX_test

for i in {1..10}
do

	echo 'Testing'  $i &>> log_riding_DRoiAlignX_test

	tools/test_net.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX/test_inference.prototxt \
		--net output/pvanet_full1_ohem_DRoiAlignX/riding_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_DRoiAlignX.yml \
		--imdb riding_2016_test  &>> log_riding_DRoiAlignX_test 
done


for i in {10..1}
do
	echo 'Testing'  $i &>> log_psdb_test_${i}
	tools/test_net.py \
	    --gpu 0 \
	    --def models/full4/test_inference.prototxt \
		--net output/pvanet_full1_ohem/psdb_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem.yml \
		--imdb psdb_2015_test  &>> log_psdb_test_${i}
done


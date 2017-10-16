for i in {10..10}
do
	echo 'Testing'  $i &>> log_aichal2_cluster_test_train_${i}
	tools/test_net_kp.py \
	    --gpu 0 \
	    --def models/kp_cluster/test_inference.prototxt \
		--net output/kpCluster/aichal_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/kpCluster.yml \
		--imdb aichal2_2017_train &> log_aichal2_cluster_test_train_${i} 
done


for i in {10..10}
do
	echo 'Testing'  $i &>> log_aichal_cluster_box_test_${i}
	tools/test_net.py \
	    --gpu 0 \
	    --def models/aichal_cluster/test_inference.prototxt \
		--net output/aichalCluster/aichal_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/aichalCluster.yml \
		--imdb aichal_2017_test &>> log_aichal_cluster_box_test_${i} 
done


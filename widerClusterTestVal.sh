for i in {10..10}
do
	echo 'Testing'  $i &>> log_wider_cluster_test_val_${i}
	tools/test_net.py \
	    --gpu 0 \
	    --def models/wider_cluster/test_inference.prototxt \
		--net output/wider_cluster/wider_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/wider_cluster.yml \
		--imdb wider_2015_val &>> log_wider_cluster_test_val_${i} 
done


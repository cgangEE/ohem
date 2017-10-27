for i in {10..10}
do
	echo 'Testing'  $i &>> log_aichal_cluster_fcn_val_${i}
	tools/test_net_kpFcn.py \
	    --gpu 0 \
	    --def models/kp_cluster_fcn/test_inference.prototxt \
		--net output/kpClusterFcn/aichal_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/kpClusterFcn.yml \
		--imdb aichal_2017_val &>> log_aichal_cluster_fcn_val_${i} 
done


for i in {10..10}
do
	echo 'Testing'  $i &>> log_aichal_cluster_cls_val_${i}
	tools/test_net_kp_cls.py \
	    --gpu 0 \
	    --def models/kp_cluster_cls/test_inference.prototxt \
		--net output/kpClusterCls/aichal_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/kpClusterCls.yml \
		--imdb aichal_2017_val &>> log_aichal_cluster_cls_val_${i} 
done


for i in {10..10}
do
	echo 'Testing'  $i &>> log_detviplV4d2_DRoiAlignX_Repool_test_${i}
	tools/test_net_repool.py \
	    --gpu 0 \
	    --def models/full1_DRoiAlignX_Repool/test.pt \
		--net output/pvanet_full1_ohem_DRoiAlignX/detviplV4d2_train/zf_faster_rcnn_iter_100000.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_DRoiAlignX_Repool.yml \
		--imdb detviplV4d2_2016_test  &>> log_detviplV4d2_DRoiAlignX_Repool_test_${i} 
done


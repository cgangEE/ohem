for i in {10..10}
do
	tools/test_net.py \
	    --gpu 0 \
	    --def models/full1_D/test_inference.prototxt \
		--net output/pvanet_full1_ohem_D_1500x2000/detviplV4d2_train/zf_faster_rcnn_iter_${i}0000_inference.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_D_1500x2000.yml \
		--imdb detviplV4d2XX_2016_test &>> log_detviplV4d2XX_D_test_1500x2000_${i} &
done


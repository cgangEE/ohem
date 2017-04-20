echo 'Begin' &> log_psdb_D_test

for i in {10..10}
do

	echo 'Testing'  $i &>> log_psdb_D_test

	tools/test_net.py \
	    --gpu 0 \
	    --def models/full4_D/test.pt \
		--net output/pvanet_full1_ohem_D/psdb_train/zf_faster_rcnn_iter_${i}0000.caffemodel \
	    --cfg cfgs/submit_160715_full_ohem_D.yml \
		--imdb psdb_2015_test  &>> log_psdb_D_test
done


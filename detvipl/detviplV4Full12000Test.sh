suffix='0000.caffemodel'
for i in {1..1}
do
	echo '>>>>>>>>>>>>Testing '$i
	tools/test_net.py \
	    --gpu 0 \
	    --def models/pvanet/full1_ohem/test.pt \
		--net detvipl_v4_12000_80000.caffemodel \
	    --cfg models/pvanet/cfgs/submit_160715_full.yml \
		--imdb detviplV4_2016_test &> log_detviplV4_test_12000 &
done


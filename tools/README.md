Tools for training, testing, and compressing Fast R-CNN networks.

### train_net.py
load caffemodel, train net, save caffemodel
take a snapshot after unnormalizing bbox 'cg_bbox_pred' layer

### cg_train_net.py

load caffemodel, train net, save caffemodel and solverstate
take a snapshot after unnormalizing bbox 'cg_bbox_pred' layer

### cg_restore_train_net.py

load solverstate, train net, save caffemodel and solverstate
take a snapshot after unnormalizing bbox 'cg_bbox_pred' layer


--------------------------------------------------------------------

### cg_train_head_net.py

load caffemodel, train net, save caffemodel and solverstate
take a snapshot after unnormalizing bbox 'pred_1' layers

### cg_restore_train_head_net.py

load solverstate, train net, save caffemodel and solverstate
take a snapshot after unnormalizing bbox 'pred_1' layers


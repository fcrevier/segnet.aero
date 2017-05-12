python Scripts/compute_bn_statistics.py Models/segnet_basic_train.prototxt Models/Training/proj_iter_$1.caffemodel Models/Inference/
mv ./Models/Inference/test_weights.caffemodel ./Models/Inference/proj_weights.caffemodel

#!/bin/bash
model=/face/zphe/aibeereidtorch/body_reid_general_res50_caffe_gpu_all-store-general_20200920_v010002/deploy.prototxt
weights=/face/zphe/aibeereidtorch/body_reid_general_res50_caffe_gpu_all-store-general_20200920_v010002/model_0089999.pth.caffemodel

merged_model=/face/zphe/aibeereidtorch/body_reid_general_res50_caffe_gpu_all-store-general_20200920_v010002/deploy_wobn.prototxt
merged_weights=/face/zphe/aibeereidtorch/body_reid_general_res50_caffe_gpu_all-store-general_20200920_v010002/model_0089999_wobn.caffemodel
input_h=384
input_w=128
GPU_ID=0
feat_blob=reid

#python convert_2_nonbnn.py \
#--model ${model} \
#--weights ${weights} \
#--merged-model ${merged_model} \
#--merged-weights ${merged_weights}

python test_convert.py \
--model ${model} \
--weights ${weights} \
--merged-model ${merged_model} \
--merged-weights ${merged_weights} \
--feat_blob ${feat_blob} \
--input_h ${input_h} \
--input_w ${input_w} \
--GPU_ID ${GPU_ID}

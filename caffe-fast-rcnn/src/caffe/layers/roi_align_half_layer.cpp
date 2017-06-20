// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"

#ifdef _MSC_VER
#define round(x) ((int)((x) + 0.5))
#endif

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIAlignHalfLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  CHECK_GT(roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_pool_param.pooled_h();
  pooled_width_ = roi_pool_param.pooled_w();
  spatial_scale_ = roi_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignHalfLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_h_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_w_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);

  bottom_rois_half_.Reshape(bottom[1]->num(), bottom[1]->channels(), 
		  bottom[1]->height(), bottom[1]->width());
}

template <typename Dtype>
void ROIAlignHalfLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	LOG(FATAL) << "Not Implemented Yet";
}

template <typename Dtype>
void ROIAlignHalfLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	LOG(FATAL) << "Not Implemented Yet";
}


#ifdef CPU_ONLY
STUB_GPU(ROIAlignHalfLayer);
#endif

INSTANTIATE_CLASS(ROIAlignHalfLayer);
REGISTER_LAYER_CLASS(ROIAlignHalf);

}  // namespace caffe

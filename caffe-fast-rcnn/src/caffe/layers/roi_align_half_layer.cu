// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"

namespace caffe {

  template <typename Dtype>
    __global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
        const Dtype spatial_scale, const int channels, const int height,
        const int width, const int pooled_height, const int pooled_width,
        const Dtype* bottom_rois, Dtype* top_data, 
        Dtype* argmax_data_h, Dtype* argmax_data_w) {

      CUDA_KERNEL_LOOP(index, nthreads) {
        // (n, c, ph, pw) 是 RoiAlign 要给 R-CNN 提取对应在 200 X 512 X 6 X 6 Featurn map 上的坐标
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        // (roi_start_h, roi_end_h, roi_start_w, roi_end_w) 是在 convf 1 X 512 X 80 X 132，也就是原图大小 1/8 的那个 feature map 上的坐标
        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
        Dtype roi_start_w = bottom_rois[1] * spatial_scale;
        Dtype roi_start_h = bottom_rois[2] * spatial_scale;
        Dtype roi_end_w = bottom_rois[3] * spatial_scale;
        Dtype roi_end_h = bottom_rois[4] * spatial_scale;

        Dtype roi_width = max(roi_end_w - roi_start_w, Dtype(1));
        Dtype roi_height = max(roi_end_h - roi_start_h, Dtype(1));
        Dtype bin_size_h = roi_height / pooled_height;
        Dtype bin_size_w = roi_width / pooled_width;

        //  (hstart, hend, wstart, wend) 是 ROI 中 6 X 6 第 ph X pw 的那一个小格对于在 convf 上的坐标
        Dtype hstart = ph * bin_size_h;
        Dtype wstart = pw * bin_size_w;
        Dtype hend = (ph + 1) * bin_size_h;
        Dtype wend = (pw + 1) * bin_size_w;

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, Dtype(0)), Dtype(height));
        hend = min(max(hend + roi_start_h, Dtype(0)), Dtype(height));
        wstart = min(max(wstart + roi_start_w, Dtype(0)), Dtype(width));
        wend = min(max(wend + roi_start_w, Dtype(0)), Dtype(width));
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        // 在 (hstart, hend, wstart, wend) 这一个小格上在 convf 上对应很多点，需要在这些点中找到一个最大值作为 (ph, pw) 这一小格的值，
        // maxval 为该最大值，(maxh, maxw) 为最大值的在 convf 上的坐标。

        Dtype maxval = is_empty ? 0 : -FLT_MAX;

        Dtype maxh = -1;
        Dtype maxw = -1;

        bottom_data += (roi_batch_ind * channels + c) * height * width;

        // 现在遍历 (h, w) 为 (hstart, hend, wstart, wend) 这个小格在 convf 上的很多点，
        // 注意 (h, w) 为浮点型，需要通过相邻的4个整数坐标来插值 (h, w) 对应坐标的值。

        for (Dtype h = hstart; h < hend; ++h){
          for (Dtype w = wstart; w < wend; ++w){

            // 越界处理 
            if (int(ceil(h)) == height)
              h = height - 1;
            if (int(ceil(w)) == width)
              w = width - 1;

            // (h1, w1), (h2, w1), (h1, w2), (h2, w2) 为 (h, w) 相邻的4个整数坐标，q11, q21, q12, q22 为这些坐标在 convf 上的值。
            int h1 = int(h);
            int h2 = int(ceil(h));
            int w1 = int(w);
            int w2 = int(ceil(w));

            Dtype q11 = bottom_data[h1 * width + w1];
            Dtype q21 = bottom_data[h2 * width + w1];
            Dtype q12 = bottom_data[h1 * width + w2];
            Dtype q22 = bottom_data[h2 * width + w2];

            // 下面是双线性插值的公式，插值保存在 val 中。
            Dtype val;

            if (h1 == h2){
              if (w1 == w2)
                val = q11;
              else
                val = q11 * (w2 - w) + q12 * (w - w1); 

            } else if (w1 == w2) {
              val = q11 * (h2 - h) + q21 * (h - h1);
            } else {
              val = q11 * (h2 - h) * (w2 - w) + 
                q12 * (h2 - h) * (w - w1) + 
                q21 * (h - h1) * (w2 - w) + 
                q22 * (h - h1) * (w - w1);
            }

            if (val > maxval) {
              maxval = val;
              maxh = h;
              maxw = w;
            }
          }
        }

        // 将最大值保存。
        top_data[index] = maxval;

        // 将最大值对应的坐标保存，以便BP时进行梯度回传。
        argmax_data_h[index] = maxh;
        argmax_data_w[index] = maxw;

      }
    }

  template <typename Dtype>
    void GetBottomRoisHalf(const Dtype* bottom_rois, 
        Dtype* bottom_rois_half, int num_) {

      for (int i=0; i<num_; ++i) {
        for (int j=0; j<5; ++j) { 
          bottom_rois_half[j] = bottom_rois[j];
        }

        bottom_rois_half[4] = (bottom_rois[2] + bottom_rois[4]) / 2;

        bottom_rois += 5;
        bottom_rois_half += 5;
      }
    }



  template <typename Dtype>
    void ROIAlignHalfLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
      const Dtype* bottom_data = bottom[0]->gpu_data();
      const Dtype* bottom_rois = bottom[1]->gpu_data();

      Dtype* top_data = top[0]->mutable_gpu_data();

      Dtype* argmax_data_h = max_idx_h_.mutable_gpu_data();
      Dtype* argmax_data_w = max_idx_w_.mutable_gpu_data();


      // _cg_ add bottom_rois_half 
      const Dtype* bottom_rois_cpu = bottom[1]->cpu_data();
      Dtype *bottom_rois_half_cpu = bottom_rois_half_.mutable_cpu_data();
      GetBottomRoisHalf(bottom_rois_cpu, bottom_rois_half_cpu, bottom[1]->num());
      const Dtype* bottom_rois_half = bottom_rois_half_.gpu_data();

      int count = top[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      ROIPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, spatial_scale_, channels_, height_, width_,
          pooled_height_, pooled_width_, bottom_rois_half, top_data, 
          argmax_data_h, argmax_data_w);
      CUDA_POST_KERNEL_CHECK;
    }

  template <typename Dtype>
    __global__ void ROIPoolBackward(const int nthreads, const Dtype* top_diff,
        const Dtype* argmax_data_h, const Dtype* argmax_data_w, 
        const int num_rois, const Dtype spatial_scale,
        const int channels, const int height, const int width,
        const int pooled_height, const int pooled_width, Dtype* bottom_diff,
        const Dtype* bottom_rois) {
      CUDA_KERNEL_LOOP(index, nthreads) {
        // (n, c, h, w) coords in bottom data
        // (n, c, h, w) 对应于 convf 1 X 512 X 80 X 132，也就是原图大小 1/8 的那个 feature map 的坐标

        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height / channels;

        Dtype gradient = 0;

        // (n, c, h, w) 这个点对应多个 ROI，将每一个 ROI 回传的梯度进行累加

        // Accumulate gradient over all ROIs that pooled this element

        for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
          const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
          int roi_batch_ind = offset_bottom_rois[0];
          // Skip if ROI's batch index doesn't match n
          if (n != roi_batch_ind) {
            continue;
          }

          // (roi_start_h, roi_end_h, roi_start_w, roi_end_w) 是在 convf 1 X 512 X 80 X 132，也就是原图大小 1/8 的那个 feature map 上的坐标
          Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
          Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
          Dtype roi_end_w = offset_bottom_rois[3] * spatial_scale;
          Dtype roi_end_h = offset_bottom_rois[4] * spatial_scale;

          // Skip if ROI doesn't include (h, w)
          const bool in_roi = (roi_start_w - 1 < w && w < roi_end_w + 1 &&
              roi_start_h - 1 < h && h < roi_end_h + 1);

          if (!in_roi) {
            continue;
          }


          // 找到 top_diff, argmax_data_h, argmax_data_h 对应 n x c x 0 x 0 的那个内存地址 

          int offset = (roi_n * channels + c) * pooled_height * pooled_width;
          const Dtype* offset_top_diff = top_diff + offset;

          const Dtype* offset_argmax_data_h = argmax_data_h + offset;
          const Dtype* offset_argmax_data_w = argmax_data_w + offset;


          // 计算 ROI 的宽和高
          Dtype roi_width = max(roi_end_w - roi_start_w, Dtype(1));
          Dtype roi_height = max(roi_end_h - roi_start_h, Dtype(1));

          // 计算 ROI 6*6 的 bin 的宽高			
          Dtype bin_size_h = roi_height / pooled_height;
          Dtype bin_size_w = roi_width / pooled_width;


          // 计算在 convf 上的坐标 (h, w) 可能 pool 到的 6*6 bin 的范围 （phstart, phend, pwstart, pwend)
          int phstart = ceil( (h - roi_start_h - 1) / bin_size_h - 1 );
          int phend = ceil( (h - roi_start_h + 1) / bin_size_h );
          int pwstart = ceil( (w - roi_start_w -1)  / bin_size_w - 1 );
          int pwend = ceil( (w - roi_start_w + 1) / bin_size_w );

          phstart = min(max(phstart, 0), pooled_height);
          phend = min(max(phend, 0), pooled_height);
          pwstart = min(max(pwstart, 0), pooled_width);
          pwend = min(max(pwend, 0), pooled_width);

          // 遍历 (phstart, phend, pwstart, pwend)
          for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {

              // 在 ph * pw 上这个bin 保存的最大值在 convf 上的坐标 (ah, aw)
              Dtype ah = offset_argmax_data_h[ph * pooled_width + pw];
              Dtype aw = offset_argmax_data_w[ph * pooled_width + pw];

              // 计算 (ah, aw) 4个相邻的坐标，看是否为 (h, w), 若是则累加相应的梯度。
              int h1 = int(ah);
              int h2 = int(ceil(ah));
              int w1 = int(aw);
              int w2 = int(ceil(aw));

              if (h1 <= h && h <= h2 && 
                  w1 <= w && w <= w2) {
                Dtype gradient_factor = 1.0;

                if (h == h1)
                  gradient_factor *= h2 - ah;
                else
                  gradient_factor *= ah - h1;

                if (w == w1)
                  gradient_factor *= w2 - aw;
                else
                  gradient_factor *= aw - w1;

                gradient += offset_top_diff[ph * pooled_width + pw] * gradient_factor;
              }
            }
          }
        }
        bottom_diff[index] = gradient;
      }
    }

  template <typename Dtype>
    void ROIAlignHalfLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      if (!propagate_down[0]) {
        return;
      }
      const Dtype* bottom_rois = bottom[1]->gpu_data();
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const int count = bottom[0]->count();
      caffe_gpu_set(count, Dtype(0.), bottom_diff);


      const Dtype* argmax_data_h = max_idx_h_.gpu_data();
      const Dtype* argmax_data_w = max_idx_w_.gpu_data();

      // _cg_ add bottom_rois_half 
      const Dtype* bottom_rois_half = bottom_rois_half_.gpu_data();

      // NOLINT_NEXT_LINE(whitespace/operators)
      ROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, argmax_data_h, argmax_data_w, top[0]->num(), spatial_scale_, channels_,
          height_, width_, pooled_height_, pooled_width_, bottom_diff, 
          bottom_rois_half);
      CUDA_POST_KERNEL_CHECK;
    }

  INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignHalfLayer);

}  // namespace caffe

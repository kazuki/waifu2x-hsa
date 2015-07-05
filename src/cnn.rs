use image::Image;
use model::Model;
use super::PerfStatus;

pub fn filter_cpu(in_img: Image, model: &Model, perf: &mut PerfStatus) -> Image {

    let mut in_maps: Vec<Vec<f32>> = Vec::new();
    let mut out_maps = in_img.data;
    let mut width = in_img.width;
    let mut width_stride = in_img.strides[0];
    let mut height = in_img.height;
    
    for layer in model.iter() {
        in_maps = out_maps;
        out_maps = Vec::new();

        let new_width = width - 2;
        let new_height = height - 2;
        
        for i in 0..layer.nOutputPlane {
            let bias = &layer.bias[i as usize];
            let weights = &layer.weight[i as usize];
            let mut out: Vec<f32> = Vec::with_capacity(new_width * new_height);
            unsafe { out.set_len(new_width * new_height); }
            for j in 0..out.len() {
                out[j] = 0.0;
            }

            for j in 0..layer.nInputPlane as usize {
                let in_map = &in_maps[j];
                let w = &weights[j];
                for y in 0..new_height {
                    let off_src0 = y * width_stride;
                    let off_src1 = (y + 1) * width_stride;
                    let off_src2 = (y + 2) * width_stride;
                    let off_dst = y * new_width;
                    for x in 0..new_width {
                        out[off_dst + x] +=
                            in_map[off_src0 + x + 0] * w[0][0] +
                            in_map[off_src0 + x + 1] * w[0][1] +
                            in_map[off_src0 + x + 2] * w[0][2] +
                            in_map[off_src1 + x + 0] * w[1][0] +
                            in_map[off_src1 + x + 1] * w[1][1] +
                            in_map[off_src1 + x + 2] * w[1][2] +
                            in_map[off_src2 + x + 0] * w[2][0] +
                            in_map[off_src2 + x + 1] * w[2][1] +
                            in_map[off_src2 + x + 2] * w[2][2];
                    }
                }
            }

            for j in 0..out.len() {
                out[j] = out[j] + bias;
                if out[j] < 0.0 {
                    out[j] = out[j] * 0.1;
                }
            }

            out_maps.push(out);
        }

        width = new_width;
        height = new_height;
        width_stride = width;
        perf.cnn_flo +=
            ((layer.nInputPlane * layer.nOutputPlane * layer.kW * layer.kH * 2) as u64 * (width * height) as u64) +
            (layer.nOutputPlane as u64 * width as u64 * height as u64);
    }

    let mut out_strides = Vec::new();
    for _ in 0..out_maps.len() {
        out_strides.push(width_stride);
    }
    Image {
        width: width,
        height: height,
        color_space: in_img.color_space.clone(),
        data: out_maps,
        strides: out_strides,
    }
}

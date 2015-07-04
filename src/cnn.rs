use model::Model;

pub fn filter_cpu(in_feature_maps: Vec<Vec<f32>>, padded_width: usize,
                  padded_height: usize, stride: usize, model: &Model) -> Vec<Vec<f32>> {

    let mut in_maps: Vec<Vec<f32>> = Vec::new();
    let mut out_maps = in_feature_maps;
    let mut width = padded_width;
    let mut width_stride = stride;
    let mut height = padded_height;
    
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
    }

    out_maps
}

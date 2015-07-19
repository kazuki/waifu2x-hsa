use std;
use image::Image;
use model::{Model, Layer};
use super::PerfStatus;

pub fn filter_cpu1(in_img: Image, model: &Model, perf: &mut PerfStatus) -> Image {

    let mut in_maps = in_img.data;
    let mut out_maps: Vec<Vec<f32>> = Vec::new();
    let mut width = in_img.width;
    let mut width_stride = in_img.strides[0];
    let mut height = in_img.height;
    
    for layer in model.iter() {
        let new_width = width - 2;
        let new_height = height - 2;

        unsafe {
            for i in 0..layer.nOutputPlane {
                let bias = &layer.bias[i as usize];
                let weights = &layer.weight[i as usize];
                let mut out: Vec<f32> = Vec::with_capacity(new_width * new_height);
                out.set_len(new_width * new_height);

                for j in 0..layer.nInputPlane as usize {
                    let in_map = in_maps.get_unchecked(j);
                    let w = weights.get_unchecked(j);
                    let w00 = *w.get_unchecked(0).get_unchecked(0);
                    let w01 = *w.get_unchecked(0).get_unchecked(1);
                    let w02 = *w.get_unchecked(0).get_unchecked(2);
                    let w10 = *w.get_unchecked(1).get_unchecked(0);
                    let w11 = *w.get_unchecked(1).get_unchecked(1);
                    let w12 = *w.get_unchecked(1).get_unchecked(2);
                    let w20 = *w.get_unchecked(2).get_unchecked(0);
                    let w21 = *w.get_unchecked(2).get_unchecked(1);
                    let w22 = *w.get_unchecked(2).get_unchecked(2);

                    for y in 0..new_height {
                        let off_src0 = y * width_stride;
                        let off_src1 = (y + 1) * width_stride;
                        let off_src2 = (y + 2) * width_stride;
                        let off_dst = y * new_width;
                        for x in 0..new_width {
                            *out.get_unchecked_mut(off_dst + x) +=
                                *in_map.get_unchecked(off_src0 + x + 0) * w00 +
                                *in_map.get_unchecked(off_src0 + x + 1) * w01 +
                                *in_map.get_unchecked(off_src0 + x + 2) * w02 +
                                *in_map.get_unchecked(off_src1 + x + 0) * w10 +
                                *in_map.get_unchecked(off_src1 + x + 1) * w11 +
                                *in_map.get_unchecked(off_src1 + x + 2) * w12 +
                                *in_map.get_unchecked(off_src2 + x + 0) * w20 +
                                *in_map.get_unchecked(off_src2 + x + 1) * w21 +
                                *in_map.get_unchecked(off_src2 + x + 2) * w22;
                        }
                    }
                }

                for j in 0..out.len() {
                    *out.get_unchecked_mut(j) = *out.get_unchecked(j) + bias;
                    if *out.get_unchecked(j) < 0.0 {
                        *out.get_unchecked_mut(j) *= 0.1;
                    }
                }
                out_maps.push(out);
            }
        }

        width = new_width;
        height = new_height;
        width_stride = width;
        perf.cnn_flo +=
            ((layer.nInputPlane * layer.nOutputPlane * layer.kW * layer.kH * 2) as u64 * (width * height) as u64) +
            (layer.nOutputPlane as u64 * width as u64 * height as u64);

        in_maps = out_maps;
        out_maps = Vec::new();
    }
    out_maps = in_maps;

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

pub fn filter_cpu2(in_img: Image, model: &Model, perf: &mut PerfStatus) -> Image {
    let stride = {
        let mut max_stride: usize = 1;
        for i in 0..model.len() as usize {
            let layer = &model[i];
            max_stride = std::cmp::max(max_stride, std::cmp::max(
                (in_img.width - i * 2) * layer.nInputPlane as usize,
                (in_img.width - (i + 1) * 2) * layer.nOutputPlane as usize));
        }
        max_stride
    };
    let mut buf = Vec::<f32>::with_capacity(stride * in_img.height);
    unsafe {
        let capacity = buf.capacity();
        buf.set_len(capacity);
    }

    {
        let cnt = model[0].nInputPlane as usize;
        for y in 0..in_img.height {
            let off = stride * y * cnt;
            for i in 0..cnt {
                let src_off = in_img.strides[i] * y;
                for x in 0..in_img.width {
                    buf[off + x * cnt + i] = in_img.data[i][src_off + x];
                }
            }
        }
    }

    let mut temp: [f32; 128] = [0.0; 128];
    let mut out_line: Vec<f32> = Vec::with_capacity(stride);
    unsafe { out_line.set_len(stride); }
    let (mut width, mut height) = (in_img.width, in_img.height);

    for layer in model.iter() {
        width -= 2;
        height -= 2;
        perf.cnn_flo +=
            ((layer.nInputPlane * layer.nOutputPlane * layer.kW * layer.kH * 2) as u64 * (width * height) as u64) +
            (layer.nOutputPlane as u64 * width as u64 * height as u64);

        if layer.nOutputPlane < 32 {
            filter_cpu2_layer(layer, layer.nInputPlane as usize, layer.nOutputPlane as usize,
                              width, height, stride, &mut temp, &mut buf, &mut out_line);
        } else {
            filter_cpu2_layer_32(layer, layer.nInputPlane as usize, layer.nOutputPlane as usize,
                                 width, height, stride, &mut temp, &mut buf, &mut out_line);
        }
    }

    let out_maps = {
        let mut out_maps = Vec::new();
        let num_out = model[model.len() - 1].nOutputPlane as usize;
        for i in 0..num_out {
            let mut v = Vec::with_capacity(width * height);
            unsafe { v.set_len(width * height); }
            for y in 0..height {
                let src_off = y * stride;
                let dst_off = y * width;
                for x in 0..width {
                    v[dst_off + x] = buf[src_off + x * num_out + i];
                }
            }
            out_maps.push(v);
        }
        out_maps
    };
    let out_strides = {
        let mut v = Vec::new();
        for _ in 0..out_maps.len() {
            v.push(width);
        }
        v
    };
    Image {
        width: width,
        height: height,
        color_space: in_img.color_space.clone(),
        data: out_maps,
        strides: out_strides,
    }
}

fn filter_cpu2_layer(layer: &Layer, num_in: usize, num_out: usize,
                     width: usize, height: usize, stride: usize,
                     temp: &mut [f32; 128], buf: &mut Vec<f32>, out_line: &mut Vec<f32>) {
    let weights = filter_cpu2_get_weights(&layer);
    let mut input: [f32; 9] = [0.0; 9];
    unsafe {
        for y in 0..height {
            for x in 0..width {
                let in_off00 = y * stride + x * num_in;
                let in_off01 = in_off00 + num_in;
                let in_off02 = in_off00 + num_in * 2;
                let in_off10 = in_off00 + stride;
                let in_off11 = in_off10 + num_in;
                let in_off12 = in_off10 + num_in * 2;
                let in_off20 = in_off00 + stride * 2;
                let in_off21 = in_off20 + num_in;
                let in_off22 = in_off20 + num_in * 2;
                for i in 0..num_in {
                    let w = weights.get_unchecked(i);
                    *input.get_unchecked_mut(0) = *buf.get_unchecked(in_off00 + i);
                    *input.get_unchecked_mut(1) = *buf.get_unchecked(in_off01 + i);
                    *input.get_unchecked_mut(2) = *buf.get_unchecked(in_off02 + i);
                    *input.get_unchecked_mut(3) = *buf.get_unchecked(in_off10 + i);
                    *input.get_unchecked_mut(4) = *buf.get_unchecked(in_off11 + i);
                    *input.get_unchecked_mut(5) = *buf.get_unchecked(in_off12 + i);
                    *input.get_unchecked_mut(6) = *buf.get_unchecked(in_off20 + i);
                    *input.get_unchecked_mut(7) = *buf.get_unchecked(in_off21 + i);
                    *input.get_unchecked_mut(8) = *buf.get_unchecked(in_off22 + i);
                    for j in 0..num_out {
                        *temp.get_unchecked_mut(j) += convolve3x3(&input, &w, j * 9);
                    }
                }
                filter_cpu2_layer_bias_relu(&layer, num_out, x, temp, out_line);
            }
            filter_cpu2_layer_writeback_line(num_out, width, y, stride, buf, out_line);
        }
    }
}

fn filter_cpu2_layer_32(layer: &Layer, num_in: usize, num_out: usize,
                        width: usize, height: usize, stride: usize,
                        temp: &mut [f32; 128], buf: &mut Vec<f32>, out_line: &mut Vec<f32>) {
    let weights = filter_cpu2_get_weights(&layer);
    let mut input: [f32; 9] = [0.0; 9];
    unsafe {
        for y in 0..height {
            for x in 0..width {
                let in_off00 = y * stride + x * num_in;
                let in_off01 = in_off00 + num_in;
                let in_off02 = in_off00 + num_in * 2;
                let in_off10 = in_off00 + stride;
                let in_off11 = in_off10 + num_in;
                let in_off12 = in_off10 + num_in * 2;
                let in_off20 = in_off00 + stride * 2;
                let in_off21 = in_off20 + num_in;
                let in_off22 = in_off20 + num_in * 2;
                for i in 0..num_in {
                    let w = weights.get_unchecked(i);
                    *input.get_unchecked_mut(0) = *buf.get_unchecked(in_off00 + i);
                    *input.get_unchecked_mut(1) = *buf.get_unchecked(in_off01 + i);
                    *input.get_unchecked_mut(2) = *buf.get_unchecked(in_off02 + i);
                    *input.get_unchecked_mut(3) = *buf.get_unchecked(in_off10 + i);
                    *input.get_unchecked_mut(4) = *buf.get_unchecked(in_off11 + i);
                    *input.get_unchecked_mut(5) = *buf.get_unchecked(in_off12 + i);
                    *input.get_unchecked_mut(6) = *buf.get_unchecked(in_off20 + i);
                    *input.get_unchecked_mut(7) = *buf.get_unchecked(in_off21 + i);
                    *input.get_unchecked_mut(8) = *buf.get_unchecked(in_off22 + i);
                    for j in (0..num_out-7).step_by(8) {
                        let wi = j * 9;
                        *temp.get_unchecked_mut(j + 0) += convolve3x3(&input, &w, wi);
                        *temp.get_unchecked_mut(j + 1) += convolve3x3(&input, &w, wi + 9);
                        *temp.get_unchecked_mut(j + 2) += convolve3x3(&input, &w, wi + 18);
                        *temp.get_unchecked_mut(j + 3) += convolve3x3(&input, &w, wi + 27);
                        *temp.get_unchecked_mut(j + 4) += convolve3x3(&input, &w, wi + 36);
                        *temp.get_unchecked_mut(j + 5) += convolve3x3(&input, &w, wi + 45);
                        *temp.get_unchecked_mut(j + 6) += convolve3x3(&input, &w, wi + 54);
                        *temp.get_unchecked_mut(j + 7) += convolve3x3(&input, &w, wi + 63);
                    }
                }
                filter_cpu2_layer_bias_relu(&layer, num_out, x, temp, out_line);
            }
            filter_cpu2_layer_writeback_line(num_out, width, y, stride, buf, out_line);
        }
    }
}

#[inline(always)]
fn convolve3x3(input: &[f32; 9], w: &Vec<f32>, off: usize) -> f32 {
    unsafe {
        *input.get_unchecked(0) * *w.get_unchecked(off) +
            *input.get_unchecked(1) * *w.get_unchecked(off + 1) +
            *input.get_unchecked(2) * *w.get_unchecked(off + 2) +
            *input.get_unchecked(3) * *w.get_unchecked(off + 3) +
            *input.get_unchecked(4) * *w.get_unchecked(off + 4) +
            *input.get_unchecked(5) * *w.get_unchecked(off + 5) +
            *input.get_unchecked(6) * *w.get_unchecked(off + 6) +
            *input.get_unchecked(7) * *w.get_unchecked(off + 7) +
            *input.get_unchecked(8) * *w.get_unchecked(off + 8)
    }
}

fn filter_cpu2_get_weights(layer: &Layer) -> Vec<Vec<f32>> {
    let mut w = Vec::with_capacity(layer.nInputPlane as usize);
    for i in 0..layer.nInputPlane as usize {
        let mut v = Vec::with_capacity(9 * layer.nOutputPlane as usize);
        for j in 0..layer.nOutputPlane as usize {
            for x in 0..3 {
                for y in 0..3 {
                    v.push(layer.weight[j][i][x][y]);
                }
            }
        }
        w.push(v);
    }
    w
}

#[inline(always)]
fn filter_cpu2_layer_bias_relu(layer: &Layer, num_out: usize, x: usize,
                               temp: &mut [f32; 128], out_line: &mut Vec<f32>) {
    unsafe {
        for i in 0..num_out {
            let mut v = *temp.get_unchecked(i) + *layer.bias.get_unchecked(i);
            if v < 0.0 {
                v *= 0.1;
            }
            *out_line.get_unchecked_mut(x * num_out + i) = v;
            *temp.get_unchecked_mut(i) = 0.0;
        }
    }
}

#[inline(always)]
fn filter_cpu2_layer_writeback_line(num_out: usize, width: usize, y: usize, stride: usize,
                                    buf: &mut Vec<f32>, out_line: &mut Vec<f32>) {
    unsafe {
        let out_off = y * stride;
        for i in 0..width * num_out {
            *buf.get_unchecked_mut(out_off + i) = *out_line.get_unchecked(i);
        }
    }
}

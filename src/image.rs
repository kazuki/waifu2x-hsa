use std;

extern crate image as piston_image;
use piston_image::GenericImage;

#[derive(Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub color_space: ColorSpace,
    pub data: Vec<Vec<f32>>,
    pub strides: Vec<usize>,
    pub padding: usize,
    pub alignment: usize,
}

#[derive(PartialEq, Clone)]
pub enum ColorSpace {
    RGB = 0,
    I444 = 1,
}

impl Image {
    pub fn from_dynamic_image(img: &piston_image::DynamicImage,
                              padding: usize, alignment: usize) -> Image {
        let rgb_img = img.to_rgb();
        let w = img.width() as usize;
        let h = img.height() as usize;
        let stride = _align(w + padding * 2, alignment);
        let stride_h = _align(h + padding * 2, alignment);
        let vec_size = stride * stride_h;
        let mut r: Vec<f32> = Vec::with_capacity(vec_size);
        let mut g: Vec<f32> = Vec::with_capacity(vec_size);
        let mut b: Vec<f32> = Vec::with_capacity(vec_size);
        unsafe {
            r.set_len(vec_size);
            g.set_len(vec_size);
            b.set_len(vec_size);
        }
        let src = rgb_img.into_raw();
        for y in 0..h {
            let off_src = y * w;
            let off_dst = (padding + y) * stride + padding;
            for x in 0..w {
                r[off_dst + x] = (src[(off_src + x) * 3 + 0] as f32) / 255.0;
                g[off_dst + x] = (src[(off_src + x) * 3 + 1] as f32) / 255.0;
                b[off_dst + x] = (src[(off_src + x) * 3 + 2] as f32) / 255.0;
            }
        }
        Image {
            width: w,
            height: h,
            color_space: ColorSpace::RGB,
            data: vec![r, g, b],
            strides: vec![stride, stride, stride],
            padding: padding,
            alignment: alignment,
        }
    }

    pub fn to_dynamic_image(&self) -> piston_image::DynamicImage {
        if self.color_space != ColorSpace::RGB {
            let mut tmp_img = self.clone();
            tmp_img.change_colorspace_rgb();
            return tmp_img.to_dynamic_image();
        }

        let mut dimg = piston_image::DynamicImage::new_rgb8(self.width as u32,
                                                            self.height as u32);
        let s0 = &self.data[0];
        let s1 = &self.data[1];
        let s2 = &self.data[2];
        for i in 0..self.height {
            let off = (i + self.padding) * self.strides[0] + self.padding;
            for j in 0..self.width {
                let r = std::cmp::min(255, std::cmp::max(0, (s0[off + j] * 255.0) as i32));
                let g = std::cmp::min(255, std::cmp::max(0, (s1[off + j] * 255.0) as i32));
                let b = std::cmp::min(255, std::cmp::max(0, (s2[off + j] * 255.0) as i32));
                dimg.put_pixel(j as u32, i as u32, piston_image::Rgba {
                    data: [r as u8, g as u8, b as u8, 0] 
                });
            }
        }
        dimg
    }

    pub fn scale2x(&self, padding: usize, alignment: usize) -> Image {
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(self.data.len());
        let stride = _align(self.width * 2 + padding * 2, alignment);
        let stride_h = _align(self.height * 2 + padding * 2, alignment);

        for v in self.data.iter() {
            let mut x: Vec<f32> = Vec::with_capacity(stride * stride_h);
            unsafe { x.set_len(stride * stride_h); }
            for y in 0..self.height {
                let off_src = (y + self.padding) * self.strides[0] + self.padding;
                let off_dst = padding * stride + y * stride * 2 + padding;
                for i in 0..self.width {
                    let t = v[off_src + i];
                    x[off_dst + i * 2 + 0] = t;
                    x[off_dst + i * 2 + 1] = t;
                    x[off_dst + stride + i * 2 + 0] = t;
                    x[off_dst + stride + i * 2 + 1] = t;
                }
            }
            data.push(x);
        }

        Image {
            width: self.width * 2,
            height: self.height * 2,
            color_space: self.color_space.clone(),
            data: data,
            strides: vec![stride, stride, stride],
            padding: padding,
            alignment: alignment,
        }
    }

    pub fn fill_padding_area(&mut self) {
        for k in 0..self.data.len() {
            let x = &mut self.data[k];
            let stride = self.strides[k];

            let off_lt = self.padding * stride + self.padding;
            let off_rt = off_lt + self.width - 1;
            let off_lb = (self.padding + self.height - 1) * stride + self.padding;
            let off_rb = off_lb + self.width - 1;

            for i in 0..self.width {
                let vt = x[off_lt + i];
                let vb = x[off_lb + i];
                for j in 0..self.padding {
                    x[j * stride + self.padding + i] = vt;
                    x[off_lb + (j + 1) * stride + i] = vb;
                }
            }

            for i in 0..self.height {
                let vl = x[off_lt + i * stride];
                let vr = x[off_rt + i * stride];
                for j in 0..self.padding {
                    x[off_lt + i * stride - j] = vl;
                    x[off_rt + i * stride + j] = vr;
                }
            }

            let vlt = x[off_lt];
            let vrt = x[off_rt];
            let vlb = x[off_lb];
            let vrb = x[off_rb];
            for i in 0..self.padding {
                for j in 0..self.padding {
                    x[off_lt - (i + 1) * stride - (j + 1)] = vlt;
                    x[off_rt - (i + 1) * stride + (j + 1)] = vrt;
                    x[off_lb + (i + 1) * stride - (j + 1)] = vlb;
                    x[off_rb + (i + 1) * stride + (j + 1)] = vrb;
                }
            }
        }
    }

    pub fn change_colorspace(&mut self, color_space: ColorSpace) {
        if self.color_space == color_space {
            return;
        }
        match color_space {
            ColorSpace::RGB => self.change_colorspace_rgb(),
            ColorSpace::I444 => self.change_colorspace_i444(),
        }
    }

    pub fn change_colorspace_rgb(&mut self) {
        if self.color_space == ColorSpace::RGB {
            return;
        }
        self._i444_to_rgb();
    }

    pub fn change_colorspace_i444(&mut self) {
        if self.color_space == ColorSpace::I444 {
            return;
        }
        self._rgb_to_i444();
    }

    fn _rgb_to_i444(&mut self) {
        let d = &mut self.data;
        for i in 0..d[0].len() {
            let r = d[0][i] * 255.0;
            let g = d[1][i] * 255.0;
            let b = d[2][i] * 255.0;
            d[0][i]  = (0.0 + (0.299 * r) + (0.587 * g) + (0.114 * b)) / 255.0;
            d[1][i] = (128.0 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)) / 255.0;
            d[2][i] = (128.0 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)) / 255.0;
        }
        self.color_space = ColorSpace::I444;
    }

    fn _i444_to_rgb(&mut self) {
        let d = &mut self.data;
        for i in 0..d[0].len() {
            let y = d[0][i] * 255.0;
            let u = d[1][i] * 255.0;
            let v = d[2][i] * 255.0;
            d[0][i] = (y + 1.402 * (v - 128.0)) / 255.0;
            d[1][i] = (y - 0.34414 * (u - 128.0) - 0.71414 * (v - 128.0)) / 255.0;
            d[2][i] = (y + 1.772 * (u - 128.0)) / 255.0;
        }
        self.color_space = ColorSpace::RGB;
    }
}

fn _align(x: usize, alignment: usize) -> usize {
    if x % alignment != 0 {
        x + (alignment - (x % alignment))
    } else {
        x
    }
}

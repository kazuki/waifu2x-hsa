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
}

#[derive(PartialEq, Clone)]
pub enum ColorSpace {
    RGB = 0,
    I444 = 1,
}

impl Image {
    pub fn from_dynamic_image(img: &piston_image::DynamicImage) -> Image {
        let rgb_img = img.to_rgb();
        let w = img.width() as usize;
        let h = img.height() as usize;
        let vec_size = w * h;
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
            let off = y * w;
            for x in 0..w {
                r[off + x] = (src[(off + x) * 3 + 0] as f32) / 255.0;
                g[off + x] = (src[(off + x) * 3 + 1] as f32) / 255.0;
                b[off + x] = (src[(off + x) * 3 + 2] as f32) / 255.0;
            }
        }
        Image {
            width: w,
            height: h,
            color_space: ColorSpace::RGB,
            data: vec![r, g, b],
            strides: vec![w, w, w],
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
            let off = i * self.strides[0];
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

    pub fn scale2x(&self) -> Image {
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(self.data.len());
        let stride = self.width * 2;
        let stride_h = self.height * 2;

        for v in self.data.iter() {
            let mut x: Vec<f32> = Vec::with_capacity(stride * stride_h);
            unsafe { x.set_len(stride * stride_h); }
            for y in 0..self.height {
                let off_src = y * self.strides[0];
                let off_dst = y * stride * 2;
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
        }
    }

    pub fn add_padding(&self, padding: usize) -> Image {
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(self.data.len());
        let mut strides = Vec::new();
        let stride = self.width + padding * 2;
        let stride_h = self.height + padding * 2;

        for v in self.data.iter() {
            let mut x: Vec<f32> = Vec::with_capacity(stride * stride_h);
            unsafe { x.set_len(stride * stride_h); }
            for y in 0..self.height {
                let off_src = y * self.strides[0];
                let off_dst = (padding + y) * stride + padding;
                for i in 0..self.width {
                    x[off_dst + i] = v[off_src + i];
                }
            }
            data.push(x);
            strides.push(stride);
        }

        let mut out = Image {
            width: stride,
            height: stride_h,
            color_space: self.color_space.clone(),
            data: data,
            strides: strides,
        };
        out.fill_padding_area(padding);
        out
    }

    fn fill_padding_area(&mut self, padding: usize) {
        for k in 0..self.data.len() {
            let x = &mut self.data[k];
            let stride = self.strides[k];

            let off_lt = padding * stride + padding;
            let off_rt = off_lt + self.width - padding * 2 - 1;
            let off_lb = (self.height - padding - 1) * stride + padding;
            let off_rb = off_lb + self.width - padding * 2 - 1;

            for i in 0..self.width - padding * 2{
                let vt = x[off_lt + i];
                let vb = x[off_lb + i];
                for j in 0..padding {
                    x[j * stride + padding + i] = vt;
                    x[off_lb + (j + 1) * stride + i] = vb;
                }
            }

            for i in 0..self.height - padding * 2 {
                let vl = x[off_lt + i * stride];
                let vr = x[off_rt + i * stride];
                for j in 0..padding {
                    x[off_lt + i * stride - (j + 1)] = vl;
                    x[off_rt + i * stride + (j + 1)] = vr;
                }
            }

            let vlt = x[off_lt];
            let vrt = x[off_rt];
            let vlb = x[off_lb];
            let vrb = x[off_rb];
            for i in 0..padding {
                for j in 0..padding {
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

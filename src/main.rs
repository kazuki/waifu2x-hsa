extern crate getopts;
extern crate rustc_serialize;
extern crate image as piston_image;

use std::str::FromStr;
use std::path::Path;
use std::fs::File;

use getopts::Options;

mod cnn;
mod model;
mod image;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut opts = Options::new();
    opts.reqopt("i", "input", "input image path (required)", "INPUT");
    opts.reqopt("o", "output", "output image path (required)", "OUTPUT");
    opts.optopt("s", "scale", "scale factor (default: 2)", "SCALE");
    opts.optopt("m", "method", "noise|scale|noise_scale (default: scale)", "METHOD");
    opts.reqopt("d", "model_dir", "model directory (required)", "DIR");
    opts.optopt("n", "noise_level", "1 or 2 (default: 1)", "LEVEL");
    opts.optflag("h", "help", "print this help menu");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => {
            println!("{}", f.to_string());
            print_usage(&args[0], opts);
            return;
        }
    };
    if matches.opt_present("h") {
        print_usage(&args[0], opts);
        return;
    }

    let in_path = matches.opt_str("i").unwrap();
    let out_path = matches.opt_str("o").unwrap();
    let model_dir = matches.opt_str("d").unwrap();
    let scale = match matches.opt_str("s") {
        Some(x) => match u32::from_str(x.as_ref()) {
            Ok(v) => v,
            Err(_) => panic!("cannot parse {} to unsigned-integer", x),
        },
        None => 2
    };
    let method = match matches.opt_str("m") {
        Some(x) => x,
        None => "scale".to_string()
    };
    let noise_level = match matches.opt_str("n") {
        Some(x) => match x.as_ref() {
            "1" | "2" => x,
            _ => panic!("unknown noise-level {}", x),
        },
        None => "1".to_string()
    };

    let img = match File::open(&in_path) {
        Ok(in_strm) => {
            piston_image::load(in_strm, path_to_image_format(&in_path)).unwrap()
        },
        _ => panic!("open error"),
    };
    let out_img_format = path_to_image_format(&out_path);

    let scale_model_path = Path::new(&model_dir).join(format!("scale{}.0x_model.json", scale));
    let noise_model_path = Path::new(&model_dir).join(format!("noise{}_model.json", noise_level));

    let scale_model = Box::new(model::load_model(&scale_model_path).unwrap());
    let noise_model = Box::new(model::load_model(&noise_model_path).unwrap());

    let out_img = match method.as_ref() {
        "scale" => {
            scale2(image::Image::from_dynamic_image(&img, scale_model.len(), 1), &scale_model)
        },
        "noise" => {
            filter(image::Image::from_dynamic_image(&img, noise_model.len(), 1), &noise_model)
        },
        "noise_scale" => {
            let mut tmp = image::Image::from_dynamic_image(&img, noise_model.len() + scale_model.len(), 1);
            tmp = filter(tmp, &noise_model);
            scale2(tmp, &scale_model)
        },
        _ => {
            panic!("unknown method \"{}\"", method);
        }
    };

    let mut out_strm = File::create(&out_path).unwrap();
    out_img.to_dynamic_image().save(&mut out_strm, out_img_format).unwrap();
}

fn scale2(mut img: image::Image, model: &model::Model) -> image::Image {
    img = img.scale2x(model.len(), 1);
    filter(img, &model)
}

fn filter(mut img: image::Image, model: &model::Model) -> image::Image {
    if model[0].nInputPlane == 1 && img.color_space != image::ColorSpace::I444 {
        img.change_colorspace(image::ColorSpace::I444);
    }

    img.fill_padding_area();

    let mut copy_components = Vec::new();
    if model.last().unwrap().nOutputPlane != 3 {
        for i in model.last().unwrap().nOutputPlane..3 {
            copy_components.push(img.data[i as usize].clone());
        }
    }

    let mut data = cnn::filter_cpu(img.data, img.width + img.padding * 2,
                                   img.height + img.padding * 2, img.strides[0], &model);
    let new_padding = img.padding - model.len();
    let h = img.height;
    let w = img.width;
    let stride = data[0].len() / (h + new_padding * 2);

    while copy_components.len() > 0 {
        let mut v = copy_components.remove(0);
        for y in 0..h + new_padding * 2 {
            let off_src = (y + (img.padding - new_padding)) * img.strides[0] + (img.padding - new_padding);
            let off_dst = y * stride;
            for x in 0..w + new_padding * 2 {
                v[off_dst + x] = v[off_src + x];
            }
        }
        v.truncate((h + new_padding * 2) * stride);
        data.push(v);
    }

    image::Image {
        width: w,
        height: h,
        color_space: img.color_space.clone(),
        data: data,
        strides: vec![stride, stride, stride],
        padding: new_padding,
        alignment: 1,
    }
}

fn path_to_image_format(path: &String) -> piston_image::ImageFormat {
    match Path::new(&path).extension().unwrap().to_str().unwrap().to_lowercase().as_ref() {
        "jpg" | "jpeg" => piston_image::ImageFormat::JPEG,
        "png" => piston_image::ImageFormat::PNG,
        "gif" => piston_image::ImageFormat::GIF,
        "webp" => piston_image::ImageFormat::WEBP,
        "bmp" => piston_image::ImageFormat::BMP,
        x => panic!("unknown file type: {} (path:{})", x, &path),
    }
}

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options]", program);
    print!("{}", opts.usage(&brief));
}

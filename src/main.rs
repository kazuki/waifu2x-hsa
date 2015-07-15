#![feature(step_by)]

extern crate getopts;
extern crate rustc_serialize;
extern crate image as piston_image;
extern crate time;

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

    let mut perf = PerfStatus {
        cnn_flo: 0,
        cnn_time: 0.0,
        other_time: 0.0,
    };

    let start = time::precise_time_s();
    let src_img = image::Image::from_dynamic_image(&img);
    perf.other_time += time::precise_time_s() - start;

    let out_img = match method.as_ref() {
        "scale" => {
            scale2(src_img, &scale_model, &mut perf)
        },
        "noise" => {
            filter(src_img, &noise_model, &mut perf)
        },
        "noise_scale" => {
            let tmp = filter(src_img, &noise_model, &mut perf);
            scale2(tmp, &scale_model, &mut perf)
        },
        _ => {
            panic!("unknown method \"{}\"", method);
        }
    };

    let total_time = time::precise_time_s() - start;

    let mut out_strm = File::create(&out_path).unwrap();
    out_img.to_dynamic_image().save(&mut out_strm, out_img_format).unwrap();

    println!("total: {:.2} [ms]", total_time * 1000.0);
    println!("cnn: {:.2} [GFLOPS], {:.2} [ms] ({:.2} G fp-ops)",
             (perf.cnn_flo as f64) / 1000000000.0 / perf.cnn_time,
             perf.cnn_time * 1000.0, perf.cnn_flo as f64 / 1000000000.0);
    println!("other: {:.2} [ms]", perf.other_time * 1000.0);
}

fn scale2(img: image::Image, model: &model::Model, perf: &mut PerfStatus) -> image::Image {
    let start = time::precise_time_s();
    let tmp = img.scale2x();
    perf.other_time += time::precise_time_s() - start;

    filter(tmp, &model, perf)
}

fn filter(mut img: image::Image, model: &model::Model, perf: &mut PerfStatus) -> image::Image {
    let mut start = time::precise_time_s();
    if model[0].nInputPlane == 1 && img.color_space != image::ColorSpace::I444 {
        img.change_colorspace(image::ColorSpace::I444);
    }
    let padded = img.add_padding(model.len());
    perf.other_time += time::precise_time_s() - start;

    start = time::precise_time_s();
    let mut output = cnn::filter_cpu2(padded, &model, perf);
    perf.cnn_time += time::precise_time_s() - start;

    if output.data.len() == 1 {
        start = time::precise_time_s();
        output.data.push(img.data[1].clone());
        output.data.push(img.data[2].clone());
        output.strides.push(img.strides[1]);
        output.strides.push(img.strides[2]);
        perf.other_time += time::precise_time_s() - start;
    }
    output
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

pub struct PerfStatus {
    pub cnn_flo: u64,
    pub cnn_time: f64,
    pub other_time: f64,
}

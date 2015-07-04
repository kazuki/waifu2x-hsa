use std::convert::AsRef;
use std::path::Path;
use std::fs::File;
use std::io::{Error,Read};

use rustc_serialize::json;

#[allow(non_snake_case)]
#[derive(RustcDecodable, Debug)]
pub struct Layer {
    pub nInputPlane: u32,
    pub nOutputPlane: u32,
    pub kW: u32,
    pub kH: u32,
    pub bias: Vec<f32>,
    pub weight: Vec<Vec<Vec<Vec<f32>>>>,
}

pub type Model = Vec<Layer>;

#[derive(Debug)]
pub enum LoadModelError {
    IOError(Error),
    DecoderError(json::DecoderError),
}

pub fn load_model<P: AsRef<Path>>(path: P) -> Result<Model, LoadModelError> {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(e) => return Err(LoadModelError::IOError(e)),
    };
    let mut s = String::new();
    match f.read_to_string(&mut s) {
        Ok(_) => (),
        Err(e) => return Err(LoadModelError::IOError(e)),
    }
    match json::decode(&s) {
        Ok(model) => Ok(model),
        Err(e) => Err(LoadModelError::DecoderError(e)),
    }
}

use tch::{Device, nn, vision, nn::Module};
use crate::model;

pub fn inference(image_path: &str) -> i64{
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let net = model::net(&vs.root(), 2);
    vs.load("weights/best_model.ot").unwrap();
    let image = vision::imagenet::load_image_and_resize224(image_path).unwrap().unsqueeze(0);
    let out = net.forward(&image.to_device(device));
    let prediction = out.argmax(1, false);

    i64::try_from(prediction).unwrap()
}
mod data;
mod model;
use kdam::tqdm;
use tch::{nn, nn::{Module, OptimizerConfig}, Device};
fn main() {
    let dataset_train = data::Dataset::new("data/sea_vs_jungle/train", true);
    let dataset_val = data::Dataset::new("data/sea_vs_jungle/val", true);
    dataset_train.print();
    dataset_val.print();

    let mut dataloader_train = data::DataLoader::new(dataset_train, 32, true);
    let mut dataloader_val = data::DataLoader::new(dataset_val, 32, false);


    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let net = model::net(&vs.root(), 5);

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    for epoch in 1..100 {
        let mut epoch_loss = 0.0;
        for (i, (images, labels)) in tqdm!(&mut dataloader_train).enumerate() {
            opt.zero_grad();
            let out = net.forward(&images.to_device(device)).to_device(Device::Cpu);
            let loss = out.cross_entropy_for_logits(&labels);
            epoch_loss = epoch_loss + f64::try_from(&loss).unwrap() * (out.size()[0] as f64);
            opt.backward_step(&loss);
        }
        epoch_loss = epoch_loss / (dataloader_train.len() as f64);

        let mut epoch_acc = 0.0;
        for (i, (images, labels)) in tqdm!(&mut dataloader_val).enumerate() {
            let out = net.forward(&images.to_device(device)).to_device(Device::Cpu);
            let acc = out.accuracy_for_logits(&labels);
            epoch_acc += f64::try_from(acc).unwrap() * (out.size()[0] as f64);
        }
        epoch_acc = epoch_acc / (dataloader_val.len() as f64) * 100.0;
        println!("epoch {epoch}: train_loss={epoch_loss:4}, test_acc={epoch_acc:4}");
    }
}

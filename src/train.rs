use crate::data::DataLoader;
use crate::model;
use kdam::{tqdm, BarExt};
use num_traits::float::Float;
use tch::{
    nn,
    nn::{Module, OptimizerConfig},
    Device,
};
use std::path::Path;
use std::fs::create_dir;

/// This function trains the model with train and val dataloaders
pub fn train_model(
    mut dataloader_train: DataLoader,
    mut dataloader_val: DataLoader,
    save_dir: &str,
) {

    if !Path::new(save_dir).is_dir() {
        create_dir(save_dir).unwrap();
    }

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let net = model::net(&vs.root(), 5);

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    let total_batch_train = dataloader_train.len_batch();
    let mut pbar = tqdm!(
        total = total_batch_train,
        position = 1,
        desc = format!("{:<8}", "Train"),
        force_refresh = true,
        ncols = 100
    );
    let total_batch_val = dataloader_val.len_batch();
    let mut pbar2 = tqdm!(
        total = total_batch_val,
        position = 2,
        desc = format!("{:<8}", "Val"),
        force_refresh = true,
        ncols = 100
    );
    let mut best_acc = 0.0;
    let mut best_loss: f64 = f64::infinity();
    for _ in tqdm!(
        1..10,
        position = 0,
        desc = format!("{:<8}", "Epoch"),
        ncols = 100
    ) {
        let mut epoch_acc_train = 0.0;
        let mut epoch_loss_train = 0.0;
        let mut running_samples = 0;
        for (i, (images, labels)) in (&mut dataloader_train).enumerate() {
            opt.zero_grad();
            let out = net
                .forward(&images.to_device(device))
                .to_device(Device::Cpu);
            let acc = out.accuracy_for_logits(&labels);
            let loss = out.cross_entropy_for_logits(&labels);
            epoch_acc_train += f64::try_from(acc).unwrap() * (out.size()[0] as f64);
            epoch_loss_train += f64::try_from(&loss).unwrap() * (out.size()[0] as f64);
            opt.backward_step(&loss);
            running_samples += out.size()[0];
            pbar.set_postfix(format!(
                "loss={:<7.4} - accuracy={:<7.4}",
                epoch_loss_train / (running_samples as f64),
                epoch_acc_train / (running_samples as f64) * 100.0
            ));
            let _ = pbar.update_to(i + 1);
        }
        epoch_acc_train = epoch_acc_train / (dataloader_train.len() as f64);
        epoch_loss_train = epoch_loss_train / (dataloader_train.len() as f64);

        let mut epoch_acc_val = 0.0;
        let mut epoch_loss_val = 0.0;

        running_samples = 0;
        for (i, (images, labels)) in (&mut dataloader_val).enumerate() {
            let out = net
                .forward(&images.to_device(device))
                .to_device(Device::Cpu);
            let loss = out.cross_entropy_for_logits(&labels);
            let acc = out.accuracy_for_logits(&labels);
            epoch_acc_val += f64::try_from(acc).unwrap() * (out.size()[0] as f64);
            epoch_loss_val += f64::try_from(&loss).unwrap() * (out.size()[0] as f64);

            running_samples += out.size()[0];
            pbar2.set_postfix(format!(
                "loss={:<7.4} - accuracy={:<7.4}",
                epoch_loss_val / (running_samples as f64),
                epoch_acc_val / (running_samples as f64) * 100.0
            ));
            let _ = pbar2.update_to(i + 1);
        }
        epoch_acc_val /= (dataloader_val.len() as f64);
        epoch_loss_val /= (dataloader_val.len() as f64);

        if epoch_loss_val < best_loss {
            best_loss = epoch_loss_val;
            best_acc = epoch_acc_val;
            vs.save(Path::new(save_dir).join("best_model.pt")).unwrap()
        }
    }
    println!("\n\n\n");
    println!(
        "Best validation loss = {best_loss:.4}, accuracy={:.4}",
        best_acc * 100.0
    );
}

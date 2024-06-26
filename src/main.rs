mod data;
mod model;
mod train;

use kdam::tqdm;
use tch::{
    nn,
    nn::{Module, OptimizerConfig},
    Device,
};
fn main() {
    println!("Loading Dataset / train");
    let dataset_train = data::Dataset::new("data/sea_vs_jungle/train", true);
    println!("Loading Dataset / val");
    let dataset_val = data::Dataset::new("data/sea_vs_jungle/val", true);
    dataset_train.print();
    dataset_val.print();

    let dataloader_train = data::DataLoader::new(dataset_train, 3, true);
    let dataloader_val = data::DataLoader::new(dataset_val, 3, false);
    println!("{}", dataloader_val.len_batch());
    train::train_model(dataloader_train, dataloader_val, "weights")



}

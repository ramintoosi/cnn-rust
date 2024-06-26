mod data;
mod model;

use tch::vision;
fn main() {
    let dataset_train = data::Dataset::new("data/sea_vs_jungle/train", true);
    let dataset_val = data::Dataset::new("data/sea_vs_jungle/val", true);
    dataset_train.print();
    dataset_val.print();

    let mut dataloader_train = data::DataLoader::new(dataset_train, 32, true);
    let mut dataloader_val = data::DataLoader::new(dataset_val, 32, false);

    for epoch in 0..5 {
        for (i, (images, labels)) in (&mut dataloader_val).enumerate() {
            println!("{:?}", i);
        }
    }
}
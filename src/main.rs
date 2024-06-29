mod data;
mod model;
mod train;
mod inference;


fn main() {

    println!("Loading Dataset / train");
    let dataset_train = data::Dataset::new("data/sea_vs_jungle/train");
    println!("Loading Dataset / val");
    let dataset_val = data::Dataset::new("data/sea_vs_jungle/val");
    dataset_train.print();
    dataset_val.print();

    let dataloader_train = data::DataLoader::new(dataset_train, 32, true);
    let dataloader_val = data::DataLoader::new(dataset_val, 32, false);
    println!("{}", dataloader_val.len_batch());
    train::train_model(dataloader_train, dataloader_val, "weights");

    let prediction = inference::inference("data/sea_vs_jungle/val/sea/001c31c29de8a9cd.jpg");

    println!("Prediction is {prediction}")

}

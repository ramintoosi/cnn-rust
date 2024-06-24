mod dataloader;

use tch::vision::imagenet;
fn main() {
    let dataset = dataloader::Dataset::new("data/sea_vs_jungle/");
    dataset.print();
}
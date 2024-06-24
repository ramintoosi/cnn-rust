mod dataloader;

fn main() {
    let dataset = dataloader::Dataset::new("data/sea_vs_jungle/");
    dataset.print();
}
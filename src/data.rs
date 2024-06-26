//! Our data loader module. The dataset root consist of train and val folders.
//! Each folder than consist of class folder and images are in the class folder
//! Similar to ImageFolderDataset in pytorch

use std::{path::Path, fs::{read_dir}};
use std::collections::HashMap;
use rand::seq::SliceRandom;
use rand::thread_rng;
use tch::{Tensor, vision};
use kdam::tqdm;

#[derive(Debug)]
enum Image {
    ImagePath(String),
    ImageTensor(Tensor)
}

/// Our dataset struct
pub struct Dataset {
    root: String,
    image_path: Vec<(i64, Tensor)>,
    class_to_idx:HashMap<String, i64>,
    idx_to_class:HashMap<i64, String>,
    total_size: usize,
}


impl <'a> Dataset {
    /// This function walks through the root folder and gathers images and creates a Dataset
    pub fn new<T: AsRef<Path>>(root: T, pre_load: bool) -> Dataset{
        let root = root.as_ref();

        let mut image_path: Vec<(i64, Tensor)> = Vec::new();
        let mut class_to_idx:HashMap<String, i64> = HashMap::new();
        let mut idx_to_class:HashMap<i64, String> = HashMap::new();

        Self::get_images_and_classes(&root,
                                     &pre_load,
                                     &mut image_path, &mut class_to_idx, &mut idx_to_class);

        Dataset {
            root: root.to_str().unwrap().to_string(),
            total_size: image_path.len(),
            image_path,
            class_to_idx,
            idx_to_class,
        }
    }

    /// In train or val datasets finds the classes and images
    fn get_images_and_classes(
        dir: &Path,
        pre_load: &bool,
        image_path: &mut Vec<(i64, Tensor)>,
        class_to_idx: &mut HashMap<String, i64>,
        idx_to_class: &mut HashMap<i64, String>)
    {

        for (class_id, root_class) in read_dir(&dir).unwrap().enumerate() {
            let root_class = root_class.unwrap().path().clone();
            if root_class.is_dir() {
                Self::get_images_in_folder(&root_class, pre_load, image_path, class_id as i64);
                let class_name_str = root_class.file_name().unwrap().to_str().unwrap().to_string();
                class_to_idx.insert(class_name_str.clone(), class_id as i64);
                idx_to_class.insert(class_id as i64, class_name_str.clone());
            }
        }
    }

    /// find images with specific extensions in class folder
    fn get_images_in_folder (dir: &Path, preload: &bool, image_path: &mut Vec<(i64, Tensor)>, class_idx:i64) {
        let valid_ext = vec!["jpg", "png", "jpeg"];
        let mut i = 0;
        for file_path in tqdm!(read_dir(&dir).unwrap()){
            let file_path = &file_path.unwrap().path().clone();
            if file_path.is_file() & valid_ext.contains(&file_path.extension().unwrap().to_str().unwrap().to_lowercase().as_str()) {
                let image_tensor = vision::imagenet::load_image_and_resize224(file_path);
                image_path.push((class_idx, image_tensor.unwrap()));
                i+=1;
                if i > 20 {break}
            }
        }
    }

    /// A simple print function for our Dataset
    pub fn print(&self) {
        println!("DATASET ({})", self.root);
        println!("Classes: {:?}", self.class_to_idx);
        println!("Size: {}", self.total_size);
        println!("sample of data\n{:?}", &self.image_path[1..3]);
    }

    fn get_item(&'a self, idx: usize) -> (&'a Tensor, i64){
        (&self.image_path[idx].1, self.image_path[idx].0.clone())
    }
}

pub struct DataLoader {
    dataset: Dataset,
    batch_size: i64,
    batch_index: i64,
    shuffle: bool
}

impl DataLoader {
    pub fn new(mut dataset: Dataset, batch_size: i64, shuffle: bool) -> DataLoader{
        // let mut rng = thread_rng();
        // dataset.ImagePath.shuffle(rng);
        DataLoader {
            dataset,
            batch_size,
            batch_index: 0,
            shuffle
        }
    }

    fn shuffle_dataset(&mut self) {
        let mut rng = thread_rng();
        self.dataset.image_path.shuffle(&mut rng)
    }

}

impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let start = (self.batch_index * self.batch_size) as usize;
        let mut end = ((self.batch_index + 1) * self.batch_size) as usize;
        if start >= self.dataset.total_size {
            self.batch_index = 0;
            return None
        }
        if end > self.dataset.total_size {
            end = self.dataset.total_size;
        }
        if (self.batch_index == 0) & self.shuffle {
            self.shuffle_dataset();
        }
        let mut images: Vec<&Tensor> = vec![];
        let mut labels: Vec<Tensor> = vec![];

        for i in start..end {
            let (image_t, label) = self.dataset.get_item(i);
            images.push(image_t);
            labels.push(Tensor::from(label))
        }
        self.batch_index += 1;
        Some((Tensor::f_stack(&images, 0).unwrap(), Tensor::f_stack(&labels, 0).unwrap()))
    }
}

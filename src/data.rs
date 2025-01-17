//! Our data loader module. The dataset root consist of train and val folders.
//! Each folder than consist of class folder and images are in the class folder
//! Similar to ImageFolderDataset in pytorch

use kdam::tqdm;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::{fs::read_dir, path::Path};
use tch::{vision, Tensor};

pub struct Dataset {
    root: String,
    image_path: Vec<(i64, String)>,
    class_to_idx: HashMap<String, i64>,
    total_size: usize,
}

impl Dataset {
    /// This function walks through the root folder and gathers images and creates a Dataset
    pub fn new<T: AsRef<Path>>(root: T) -> Dataset {
        let root = root.as_ref();

        let mut image_path: Vec<(i64, String)> = Vec::new();
        let mut class_to_idx: HashMap<String, i64> = HashMap::new();

        Self::get_images_and_classes(
            &root,
            &mut image_path,
            &mut class_to_idx,
        );

        Dataset {
            root: root.to_str().unwrap().to_string(),
            total_size: image_path.len(),
            image_path,
            class_to_idx,
        }
    }

    /// In the input folder finds the classes and images
    fn get_images_and_classes(
        dir: &Path,
        image_path: &mut Vec<(i64, String)>,
        class_to_idx: &mut HashMap<String, i64>,
    ) {
        for (class_id, root_class) in read_dir(&dir).unwrap().enumerate() {
            let root_class = root_class.unwrap().path().clone();
            if root_class.is_dir() {
                Self::get_images_in_folder(&root_class, image_path, class_id as i64);
                let class_name_str = root_class
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();
                class_to_idx.insert(class_name_str.clone(), class_id as i64);
            }
        }
    }

    /// find images with specific extensions "jpg", "png", "jpeg"
    fn get_images_in_folder(
        dir: &Path,
        image_path: &mut Vec<(i64, String)>,
        class_idx: i64,
    ) {
        let valid_ext = vec!["jpg", "png", "jpeg"];
        for file_path in tqdm!(read_dir(&dir).unwrap()) {
            let file_path = &file_path.unwrap().path().clone();
            if file_path.is_file()
                & valid_ext.contains(
                &file_path
                    .extension()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_lowercase()
                    .as_str(),
            )
            {
                image_path.push((class_idx, file_path.to_str().unwrap().to_string()));
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

    /// load the image into a tensor and return (image, label)
    fn get_item(&self, idx: usize) -> (Tensor, i64) {
        let image =vision::imagenet::load_image_and_resize224(&self.image_path[idx].1).unwrap();
        (image, self.image_path[idx].0.clone())
    }
}

/// A struct for our data loader
pub struct DataLoader {
    dataset: Dataset,
    batch_size: i64,
    batch_index: i64,
    shuffle: bool,
}

impl DataLoader {
    pub fn new(dataset: Dataset, batch_size: i64, shuffle: bool) -> DataLoader {
        // let mut rng = thread_rng();
        // dataset.ImagePath.shuffle(rng);
        DataLoader {
            dataset,
            batch_size,
            batch_index: 0,
            shuffle,
        }
    }

    fn shuffle_dataset(&mut self) {
        let mut rng = thread_rng();
        self.dataset.image_path.shuffle(&mut rng)
    }

    /// total number of images in the dataset
    pub fn len(&self) -> usize {
        self.dataset.total_size
    }

    /// number of batches based on the dataset size and batch size
    pub fn len_batch(&self) -> usize {
        (self.dataset.total_size / self.batch_size as usize) + 1
    }
}

/// implement iterator for our Dataloader to get batches of images and labels
impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let start = (self.batch_index * self.batch_size) as usize;
        let mut end = ((self.batch_index + 1) * self.batch_size) as usize;
        if start >= self.dataset.total_size {
            self.batch_index = 0;
            return None;
        }
        if end > self.dataset.total_size {
            end = self.dataset.total_size;
        }
        if (self.batch_index == 0) & self.shuffle {
            self.shuffle_dataset();
        }
        let mut images: Vec<Tensor> = vec![]; // for preload change this to Vec<&Tensor>
        let mut labels: Vec<Tensor> = vec![];
        for i in start..end {
            let (image_t, label) = self.dataset.get_item(i);
            images.push(image_t);
            labels.push(Tensor::from(label))
        }
        self.batch_index += 1;
        Some((
            Tensor::f_stack(&images, 0).unwrap(),
            Tensor::f_stack(&labels, 0).unwrap(),
        ))
    }
}

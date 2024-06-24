//! Our data loader module. The dataset root consist of train and val folders.
//! Each folder than consist of class folder and images are in the class folder
//! Similar to ImageFolderDataset in pytorch

use std::{path::Path, fs::{read_dir}};
use std::collections::HashMap;

/// Our dataset struct
pub struct Dataset {
    root: String,
    image_path_train: Vec<(usize, String)>,
    image_path_val: Vec<(usize, String)>,
    class_to_idx:HashMap<String, usize>,
    idx_to_class:HashMap<usize, String>
}


impl Dataset {
    /// This function walks through the root folder and gathers images and creates a Dataset
    pub fn new<T: AsRef<Path>>(root: T) -> Dataset{
        let root = root.as_ref();
        let binding = root.join("train");
        let root_train = binding.as_path();
        let binding = root.join("val");
        let root_val = binding.as_path();

        let mut image_path_train: Vec<(usize, String)> = Vec::new();
        let mut image_path_val: Vec<(usize, String)> = Vec::new();
        let mut class_to_idx:HashMap<String, usize> = HashMap::new();
        let mut idx_to_class:HashMap<usize, String> = HashMap::new();

        Self::get_images_and_classes(&root_train, &mut image_path_train, &mut class_to_idx, &mut idx_to_class);
        Self::get_images_and_classes(&root_val, &mut image_path_val, &mut class_to_idx, &mut idx_to_class);

        Dataset {
            root: root.to_str().unwrap().to_string(),
            image_path_train,
            image_path_val,
            class_to_idx,
            idx_to_class,
        }
    }

    /// In train or val datasets finds the classes and images
    fn get_images_and_classes(
        dir: &Path,
        image_path: &mut Vec<(usize, String)>,
        class_to_idx: &mut HashMap<String, usize>,
        idx_to_class: &mut HashMap<usize, String>)
    {

        for (class_id, root_class) in read_dir(&dir).unwrap().enumerate() {
            let root_class = root_class.unwrap().path().clone();
            if root_class.is_dir() {
                Self::get_images_in_folder(&root_class, image_path, class_id);
                let class_name_str = root_class.file_name().unwrap().to_str().unwrap().to_string();
                class_to_idx.insert(class_name_str.clone(), class_id);
                idx_to_class.insert(class_id, class_name_str.clone());
            }
        }
    }

    /// find images with specific extensions in class folder
    fn get_images_in_folder (dir: &Path, image_path: &mut Vec<(usize, String)>, class_idx:usize) {
        let valid_ext = vec!["jpg", "png", "jpeg"];
        for file_path in read_dir(&dir).unwrap(){
            let file_path = &file_path.unwrap().path().clone();
            if file_path.is_file() & valid_ext.contains(&file_path.extension().unwrap().to_str().unwrap().to_lowercase().as_str()) {
                image_path.push((class_idx, file_path.to_str().unwrap().to_string()));
            }
        }
    }

    /// A simple print function for our Dataset
    pub fn print(&self) {
        println!("DATASET ({})", self.root);
        println!("Classes: {:?}", self.class_to_idx);
        println!("sample of train images\n{:?}", &self.image_path_train[1..3]);
        println!("sample of val images\n{:?}", &self.image_path_val[1..3]);

    }
}

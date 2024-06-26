use tch::{nn, nn::Module, nn::OptimizerConfig, Device, nn::ModuleT, Tensor, vision::imagenet};

pub fn net(vs: &nn::Path, n_class: i64) -> impl Module {
    nn::seq()
        .add(nn::conv2d(vs, 3, 16, 8, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 16, 32, 4, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.view([-1, 1465472]))
        .add(nn::linear(vs / "layer1", 1465472, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer2", 128, n_class, Default::default()))
}
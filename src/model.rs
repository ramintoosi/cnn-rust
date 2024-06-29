use tch::{nn, nn::Module};

pub fn net(vs: &nn::Path, n_class: i64) -> impl Module {
    nn::seq()
        .add(nn::conv2d(vs, 3, 16, 16, Default::default()))
        .add_fn(|xs| xs.max_pool2d_default(4))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 16, 64, 4, Default::default()))
        .add_fn(|xs| xs.max_pool2d_default(2))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 64, 128, 4, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.flat_view())
        .add(nn::linear(vs, 56448 , 1024, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 1024, n_class, Default::default()))
}

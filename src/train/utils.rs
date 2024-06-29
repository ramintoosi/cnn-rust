use num_traits::Float;
use tch::nn::Optimizer;

pub struct Scheduler <'a> {
    pub opt: &'a mut Optimizer,
    patience: i64,
    factor: f64,
    lr: f64,
    step: i64,
    last_val: f64
}

impl Scheduler<'_> {
    pub fn new(
        opt: &mut Optimizer,
        mut patience: i64,
        lr: f64,
        mut factor: f64,
    ) -> Scheduler {

        if patience < 0 {patience=5}
        if factor < 0.0 {factor=0.95}

        Scheduler{
            opt,
            patience,
            factor,
            lr,
            step: 0,
            last_val: f64::infinity()
        }
    }
    /// Check the input value
    /// If we waited enough, decrease the lr
    pub fn step(&mut self, value: f64) {
        if value < self.last_val {
            self.last_val = value;
            self.step=0
        }
        else {
            self.step += 1;
            if self.step == self.patience {
                self.step = 0;
                self.lr = self.factor * self.lr;
                self.opt.set_lr(self.lr);
            }
        }
    }

    pub fn get_lr(&self) -> f64{
        self.lr
    }
}

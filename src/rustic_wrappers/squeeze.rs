use super::*;

impl<const A: usize, const B: usize> R2Tensor<A, B> {
    pub fn squeeze(&self) -> R1Tensor<B> {
        if A != 1 {
            panic!("Can't squeeze tensor of dim 0 greater than 1")
        }
        R1Tensor::<B> {
            tensor: self.tensor.squeeze(),
        }
    }
}

impl<const A: usize, const B: usize, const C: usize> R3Tensor<A, B, C> {
    pub fn squeeze(&self) -> R2Tensor<B, C> {
        if A != 1 {
            panic!("Can't squeeze tensor of dim 0 greater than 1")
        }
        R2Tensor::<B, C> {
            tensor: self.tensor.squeeze(),
        }
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> R4Tensor<A, B, C, D> {
    pub fn squeeze(&self) -> R3Tensor<B, C, D> {
        if A != 1 {
            panic!("Can't squeeze tensor of dim 0 greater than 1")
        }
        R3Tensor::<B, C, D> {
            tensor: self.tensor.squeeze(),
        }
    }
}

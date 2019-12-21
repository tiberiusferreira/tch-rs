use crate::vision::image;
use crate::Tensor;
use std::path::Path;
pub mod from_tensor;
pub mod rust_image;
mod squeeze;

/// Generic Tensors, only constrained by its rank
pub struct R1TensorGeneric {
    pub tensor: Tensor,
}

pub struct R2TensorGeneric {
    pub tensor: Tensor,
}

pub struct R3TensorGeneric {
    pub tensor: Tensor,
}

pub struct R4TensorGeneric {
    pub tensor: Tensor,
}

/// Tensors which carry its shape as type parameters
pub struct R1Tensor<const A: usize> {
    pub tensor: Tensor,
}

pub struct R2Tensor<const A: usize, const B: usize> {
    pub tensor: Tensor,
}

pub struct R3Tensor<const A: usize, const B: usize, const C: usize> {
    pub tensor: Tensor,
}

// TODO What if I want a R4 tensor with only B known?
pub struct R4Tensor<const A: usize, const B: usize, const C: usize, const D: usize> {
    pub tensor: Tensor,
}

//impl<const A: usize, const B: usize>  R2Tensor<A, B>{
//    fn transpose(&self) -> R2Tensor<B, A>{
//        R2Tensor::<B, A> {
//            tensor: self.tensor.transpose(0, 1),
//        }
//    }
//}
//
//type Foo<const N: usize> = [i32; N + 1];

//impl<const A: usize, const B: usize,  const C: usize, const D: usize, const E: usize, const F: usize>  R3Tensor<A, B, C>{
//
//    fn transpose(&self) -> R3Tensor<D, E, F>{
//        R2Tensor::<D, E, F> {
//            tensor: self.tensor.transpose(0, 1),
//        }
//    }
//}

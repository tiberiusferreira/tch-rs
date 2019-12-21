use super::*;

/// Generic Tensors, only constrained by its rank
impl From<Tensor> for R1TensorGeneric {
    fn from(tensor: Tensor) -> Self {
        tensor.size1().expect("Tensor was not of rank 1");
        R1TensorGeneric { tensor }
    }
}

impl From<Tensor> for R2TensorGeneric {
    fn from(tensor: Tensor) -> Self {
        tensor.size2().expect("Tensor was not of rank 1");
        R2TensorGeneric { tensor }
    }
}

impl From<Tensor> for R3TensorGeneric {
    fn from(tensor: Tensor) -> Self {
        tensor.size3().expect("Tensor was not of rank 3");
        R3TensorGeneric { tensor }
    }
}

impl From<Tensor> for R4TensorGeneric {
    fn from(tensor: Tensor) -> Self {
        tensor.size4().expect("Tensor was not of rank 1");
        R4TensorGeneric { tensor }
    }
}

/// Tensors with shape data as type parameters
impl<const A: usize> From<Tensor> for R1Tensor<A> {
    fn from(tensor: Tensor) -> Self {
        let a = tensor.size1().unwrap();
        assert_eq!(a as usize, A);
        R1Tensor { tensor }
    }
}

impl<const A: usize, const B: usize> From<Tensor> for R2Tensor<A, B> {
    fn from(tensor: Tensor) -> Self {
        let (a, b) = tensor.size2().unwrap();
        assert_eq!(a as usize, A);
        assert_eq!(b as usize, B);
        R2Tensor { tensor }
    }
}

impl<const A: usize, const B: usize, const C: usize> From<Tensor> for R3Tensor<A, B, C> {
    fn from(tensor: Tensor) -> Self {
        let (a, b, c) = tensor.size3().unwrap();
        assert_eq!(a as usize, A);
        assert_eq!(b as usize, B);
        assert_eq!(c as usize, C);
        R3Tensor { tensor }
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> From<Tensor>
    for R4Tensor<A, B, C, D>
{
    fn from(tensor: Tensor) -> Self {
        let (a, b, c, d) = tensor.size4().unwrap();
        assert_eq!(a as usize, A);
        assert_eq!(b as usize, B);
        assert_eq!(c as usize, C);
        assert_eq!(d as usize, D);
        R4Tensor { tensor }
    }
}

use crate::Tensor;
use crate::vision::image;
use std::path::Path;

pub struct R2Tensor<const A: usize, const B: usize> {
    pub tensor: Tensor,
}

pub struct R3Tensor<const A: usize, const B: usize, const C: usize> {
    pub tensor: Tensor,
}

pub struct R4Tensor<const A: usize, const B: usize, const C: usize, const D: usize> {
    pub tensor: Tensor,
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

impl<const A: usize, const B: usize, const C: usize> From<Tensor> for R3Tensor<A, B, C> {
    fn from(tensor: Tensor) -> Self {
        let (a, b, c) = tensor.size3().unwrap();
        assert_eq!(a as usize, A);
        assert_eq!(b as usize, B);
        assert_eq!(c as usize, C);
        R3Tensor::<A, B, C> { tensor }
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
        R4Tensor::<A, B, C, D> { tensor }
    }
}

pub fn r_load_img_as_tensor<T: AsRef<Path>, const C: usize, const W: usize, const H: usize>(
    path: T,
) -> R3Tensor<C, W, H> {
    let img_as_tensor = image::load(path).unwrap();
    let (channels, width, height) = img_as_tensor.size3().unwrap();
    assert_eq!(channels as usize, C);
    assert_eq!(width as usize, W);
    assert_eq!(height as usize, H);
    R3Tensor::<C, W, H> {
        tensor: img_as_tensor,
    }
}
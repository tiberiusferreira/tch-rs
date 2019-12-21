use super::*;

pub fn load_img_as_tensor<T: AsRef<Path>, const C: usize, const W: usize, const H: usize>(
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

pub fn load_img_as_tensor_generic<T: AsRef<Path>>(path: T) -> R3TensorGeneric {
    let img_as_tensor = image::load(path).unwrap();
    img_as_tensor.into()
}

pub fn resize_img<
    'a,
    T: Into<&'a R3TensorGeneric>,
    const C: usize,
    const W: usize,
    const H: usize,
>(
    image_tensor: T,
) -> R3Tensor<C, W, H> {
    let resized_raw_tensor =
        image::resize(&(image_tensor.into().tensor), W as i64, H as i64).unwrap();
    resized_raw_tensor.into()
}

pub fn resize_img_generic<'a, T: Into<&'a R3TensorGeneric>>(
    image_tensor: T,
    width: u32,
    height: u32,
) -> R3TensorGeneric {
    let resized_raw_tensor =
        image::resize(&(image_tensor.into().tensor), width as i64, height as i64).unwrap();
    resized_raw_tensor.into()
}

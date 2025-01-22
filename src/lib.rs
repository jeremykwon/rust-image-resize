use base64::{self, engine::general_purpose, Engine};
use image::{
    imageops::FilterType, load_from_memory, GenericImageView, ImageBuffer, ImageOutputFormat, Rgba,
};
use imageproc::filter::filter3x3;
use opencv::{
    core::{Mat, Size},
    imgcodecs::{imdecode, imencode, IMREAD_COLOR},
    imgproc::{gaussian_blur, median_blur},
    prelude::*,
    types::VectorOfu8,
};
use std::io::Cursor;
use wasm_bindgen::prelude::*;

// 이미지 사이즈를 받아 리사이즈 하는 함수
#[wasm_bindgen]
pub fn resize_image(input: &[u8], width: u32, height: u32) -> Vec<u8> {
    // 입력 이미지를 메모리에서 읽습니다.
    let input_img = image::load_from_memory(input).expect("Failed to load image");

    // 이미지 리사이징
    let resized = input_img.resize_exact(width, height, image::imageops::FilterType::Lanczos3);

    // 메모리에 결과를 JPEG 형식으로 저장
    let mut buffer = Vec::new();
    resized
        .write_to(&mut Cursor::new(&mut buffer), ImageOutputFormat::Png)
        .expect("Failed to write image");

    buffer
}

#[wasm_bindgen]
pub fn upscale_image(base64_input: &str, scale: f64) -> String {
    // Base64 문자열 디코딩
    let img_data = general_purpose::STANDARD
        .decode(base64_input)
        .expect("Failed to decode base64 image");
    let img = image::load_from_memory(&img_data).expect("Failed to load image");

    // 이미지 크기 계산
    let (width, height) = img.dimensions(); // GenericImageView에서 제공
    let new_width: u32 = (width as f64 * scale) as u32;
    let new_height: u32 = (height as f64 * scale) as u32;

    // 이미지 업스케일링
    let resized_img: image::DynamicImage =
        img.resize_exact(new_width, new_height, FilterType::Lanczos3);

    // PNG로 변환 후 Base64로 인코딩
    let mut output_data = Vec::new();

    // Vec<u8>을 Cursor로 감싸서 write_to에 전달
    let mut cursor: Cursor<&mut Vec<u8>> = std::io::Cursor::new(&mut output_data);

    resized_img
        .write_to(&mut cursor, ImageOutputFormat::Png)
        .expect("Failed to write image");

    general_purpose::STANDARD.encode(&output_data)
}

#[wasm_bindgen]
pub fn apply_sharpen_filter(base64_input: &str) -> String {
    // Base64 문자열 디코딩
    let img_data = general_purpose::STANDARD
        .decode(base64_input)
        .expect("Failed to decode base64 image");

    // 이미지 로드
    let img = load_from_memory(&img_data).expect("Failed to load image");
    let rgba_image = img.to_rgba8();

    // 3x3 샤프닝 커널 정의
    let kernel: [f32; 9] = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];

    // 샤프닝 필터 적용
    let sharpened_img = filter3x3(&rgba_image, &kernel);

    // PNG로 변환
    let mut output_data = Vec::new();
    let mut cursor = Cursor::new(&mut output_data);
    let _ = ImageBuffer::<Rgba<u8>, _>::from_raw(
        sharpened_img.width(),
        sharpened_img.height(),
        sharpened_img.into_raw(),
    )
    .expect("Failed to create image")
    .write_to(&mut cursor, image::ImageOutputFormat::Png);

    // Base64로 인코딩하여 반환
    general_purpose::STANDARD.encode(&output_data)
}

#[wasm_bindgen]
pub fn remove_noise(base64_input: &str, blur_type: &str, kernel_size: i32) -> String {
    // 1. Base64 문자열 디코딩
    let img_data = general_purpose::STANDARD
        .decode(base64_input)
        .expect("Failed to decode base64 image");

    // 2. OpenCV로 이미지 로드
    let img = Mat::from_slice(&img_data).expect("Failed to create Mat from slice");
    let mut img_mat = imdecode(&img, IMREAD_COLOR).expect("Failed to decode image");

    // 3. 노이즈 제거 필터 적용
    let mut output = Mat::default();
    match blur_type {
        "gaussian" => {
            gaussian_blur(
                &img_mat,
                &mut output,
                Size::new(kernel_size, kernel_size),
                0.0,
                0.0,
                opencv::core::BORDER_DEFAULT,
            )
            .expect("Gaussian Blur failed");
        }
        "median" => {
            median_blur(&img_mat, &mut output, kernel_size).expect("Median Blur failed");
        }
        _ => panic!("Invalid blur type! Use 'gaussian' or 'median'."),
    }

    // 4. OpenCV에서 PNG로 인코딩
    let mut encoded = VectorOfu8::new();
    imencode(".png", &output, &mut encoded).expect("Failed to encode image");

    // 5. Base64로 변환
    general_purpose::STANDARD.encode(&encoded.to_vec())
}

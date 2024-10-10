// src/lib.rs

pub mod data;
pub mod dqrls;
pub mod errors;
pub mod fitters;
pub mod linear_model;
pub mod real_matrix;

pub use data::Data;
pub use real_matrix::RealMatrix;

pub fn check_that_x_is_a_2d_matrix(x: &RealMatrix) {
    assert_eq!(x.ndim(), 2);
}

pub fn extract_dimensions_of_a_2d_matrix(x: &RealMatrix) -> (usize, usize) {
    (x.shape()[0], x.shape()[1])
}

pub fn check_that_x_and_y_have_the_same_number_of_rows(x: &RealMatrix, y: &RealMatrix) {
    assert_eq!(x.shape()[0], y.shape()[0]);
}

pub fn check_that_2d_matrix_x_is_numeric(x: &RealMatrix) {
    assert!(x.values.iter().all(|&v| v.is_finite()));
}

pub fn initialize_qr_decomposition(_q: RealMatrix, _r: RealMatrix) {
    todo!()
}

pub fn initialize_coefficients_vector(n: usize) -> RealMatrix {
    RealMatrix::with_shape(n, 1)
}

pub fn initialize_residuals_vector(n: usize) -> RealMatrix {
    RealMatrix::with_shape(n, 1)
}

pub fn initialize_effects_vector(n: usize) -> RealMatrix {
    RealMatrix::with_shape(n, 1)
}

pub fn initialize_pivots_vector(n: usize) -> RealMatrix {
    RealMatrix::with_shape(n, 1)
}

pub fn initialize_qr_auxiliary_matrix(n: usize) {
    let q = RealMatrix::with_shape(n, n);
    let r = RealMatrix::with_shape(n, n);
    initialize_qr_decomposition(q, r)
}

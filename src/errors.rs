// src/errors.rs

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LeastSquaresError {
    #[error("Matrix dimensions mismatch: expected ({expected_rows}, {expected_cols}), found ({found_rows}, {found_cols})")]
    DimensionMismatch {
        expected_rows: usize,
        expected_cols: usize,
        found_rows: usize,
        found_cols: usize,
    },
    #[error("Failed to convert RealMatrix to slice")]
    SliceConversionFailure,
    #[error("Null pointer detected during FFI call")]
    NullPointer,
    #[error("Unknown error occurred in Fortran function call")]
    Unknown,
}

#[derive(Debug, Error)]
pub enum LmFitterError {
    #[error("Matrix dimensions mismatch: expected ({expected_rows}, {expected_cols}), found ({found_rows}, {found_cols})")]
    DimensionMismatch {
        expected_rows: usize,
        expected_cols: usize,
        found_rows: usize,
        found_cols: usize,
    },
    #[error("Failed to convert RealMatrix to slice")]
    SliceConversionFailure,
    #[error("Null pointer detected during FFI call")]
    NullPointer,
    #[error("Unknown error occurred in Fortran function call")]
    Unknown,
    #[error("Failed to allocate memory for Fortran arrays")]
    MemoryAllocationFailure,
}

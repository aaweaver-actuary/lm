//! This module contains the implementation of the QR decomposition model fitter.
//! It is the primary implementation of the `FitModel` trait for the `LinearModelFitter` enum,
//! and is used to fit a linear model to a dataset using ordinary least squares (OLS) with
//! the QR decomposition method.
//!
//! # Structs
//!
//! ## `QrDecompositionFitter`
//!
//! A struct that implements the `FitModel` trait for fitting a linear model to a dataset using
//! the QR decomposition method.
//!
//! ### Public Fields
//!
//! * `data`: A zero-copy struct that holds the data for the linear model.
//! * `tol`: An optional tolerance value for determining the rank of the matrix.
//!
//! ### Private Fields
//!
//! * `beta_array`: A pre-allocated array for storing the working solution vector.
//! * `residual_array`: A pre-allocated array for storing the residuals.
//! * `q_transposed_times_y`: A pre-allocated array for storing the Q-transposed times Y matrix. If Q
//!    originally has dimensions (n x n), and Y has dimensions (n x m), the result of the matrix
//!    multiplication is a matrix with dimensions (n x m), so the pre-allocated array has
//!    (n x m) elements.
//!
//! ### Methods
//!
//! * `new(data: &'a Data, tol: Option<Tolerance>) -> Self`: Create a new instance of the
//!   `QrDecompositionFitter` struct.
//! * `perform_qr_decomposition(&mut self) -> Result<LeastSquaresReturn, LeastSquaresError>`:

// src/fitters/qr_decomposition_fitter.rs

use super::fit::FitModel;
use crate::errors::LmFitterError;
// use crate::fortran::dqrls::FortranDqrls;
use crate::types::{Data, RealMatrix, Tolerance};
use derive_builder::Builder;

#[derive(Debug, Builder)]
pub struct QrDecompositionFitter<'a> {
    pub data: &'a Data,
    pub tol: Tolerance,
}

impl<'a> QrDecompositionFitter<'a> {
    /// Return a new instance of the `QrDecompositionFitter` struct. Takes a reference to the data
    /// for the linear model and an optional tolerance value for determining the rank of the matrix.
    /// If no tolerance value is provided, the default value is used.
    ///
    /// # Example
    /// ```
    /// use lm::{fitters::QrDecompositionFitter, Data, RealMatrix};
    /// use lm::types::Tolerance;
    ///
    /// let x = RealMatrix::with_shape(3, 2);
    /// let y = RealMatrix::with_shape(3, 1);
    /// let data = Data::new(x, y);
    /// let fitter = QrDecompositionFitter::new(&data, Some(Tolerance(1e-5)));
    ///
    /// assert_eq!(fitter.x(), &x);
    /// assert_eq!(fitter.y(), &y);
    /// assert_eq!(fitter.tol(), 1e-5);
    /// ```
    pub fn new(data: &'a Data, tol: Option<Tolerance>) -> Self {
        Self {
            data,
            tol: tol.unwrap_or_default(),
            // Pre-allocate arrays for the solution, residuals, and Q-transposed times Y

            /*             // 1 element per column in X (eg per parameter)
            beta_array: vec![0.0; data.x().n_cols()],

            // 1 element per row in X (eg per observation)
            residual_array: vec![0.0; data.x().n_rows()],

            // 1 element per row in X (eg per observation) * number of columns in Y
            q_transposed_times_y: vec![0.0; data.x().n_rows() * data.y().n_cols()], */
        }
    }

    /// Return the x matrix from the data struct.
    pub fn x(&self) -> &RealMatrix {
        &self.data.x
    }

    /// Return the y matrix from the data struct.
    pub fn y(&self) -> &RealMatrix {
        &self.data.y
    }

    /// Return the unwrapped tolerance value. If no tolerance value is provided, the default value
    /// is used.
    pub fn tol(&self) -> f64 {
        self.tol.value()
    }

    /*     /// Return the solution vector from the QR decomposition.
    fn beta(&self) -> &Vec<f64> {
        &self.beta_array
    }

    /// Return the residuals from the QR decomposition.
    fn resid(&self) -> &Vec<f64> {
        &self.residual_array
    }

    /// Return the Q-transposed times Y matrix from the QR decomposition.
    fn qty(&self) -> &Vec<f64> {
        &self.q_transposed_times_y
    } */
}

impl<'a> FitModel for QrDecompositionFitter<'a> {
    /// Use the QR decomposition method to fit the linear model to the data.
    /// Calls the FFI function that wraps the LINPACK dqrls subroutine below.
    /// The function signature is:
    /// ```fortran
    /// subroutine dqrls(x,n,p,y,ny,tol,b,rsd,qty,k,jpvt,qraux,work)
    /// ```g
    ///
    /// # Returns
    ///
    fn fit(&self) -> Result<RealMatrix, LmFitterError> {
        // let result = self.perform_qr_decomposition();
        struct Temp {
            beta: RealMatrix,
        }

        let result: Result<Temp, LmFitterError> = Ok(Temp {
            beta: RealMatrix::with_shape(1, 1),
        });
        match result {
            Ok(fitted) => Ok(fitted.beta),
            Err(_e) => Err(LmFitterError::Unknown),
        }
    }

    fn x(&self) -> &RealMatrix {
        self.data.x()
    }

    fn y(&self) -> &RealMatrix {
        self.data.y()
    }
}

#[derive(Debug, Builder)]
pub struct FortranLeastSquaresReturn {
    /// `beta` is a matrix of real numbers representing the coefficients
    /// in a linear model. It is used to store the estimated parameters
    /// after fitting the model to the data.
    pub beta: RealMatrix,
    pub residuals: RealMatrix,
    pub q_transposed_times_y: RealMatrix,
    pub qr_decomp_auxiliary_information: RealMatrix,
}

impl FortranLeastSquaresReturn {
    pub fn builder() -> FortranLeastSquaresReturnBuilder {
        FortranLeastSquaresReturnBuilder::default()
    }
}

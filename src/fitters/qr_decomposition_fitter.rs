// src/fitters/qr_decomposition_fitter.rs

use super::fit::FitLinearModel;
use crate::errors::{FortranLeastSquaresError, LmFitterError};
use crate::linear_model::LinearModel;
use crate::{Data, RealMatrix};
use derive_builder::Builder;
use libc::c_int;

#[derive(Debug)]
pub struct QrDecompositionFitter<'a> {
    data: &'a Data,
    lm: LinearModel<'a>,
}

impl<'a> QrDecompositionFitter<'a> {
    pub fn new(data: &'a Data) -> Self {
        QrDecompositionFitter {
            data,
            lm: LinearModel::new(data),
        }
    }

    pub fn x(&self) -> &RealMatrix {
        &self.data.x
    }

    pub fn y(&self) -> &RealMatrix {
        &self.data.y
    }
}

impl<'a> FitLinearModel for QrDecompositionFitter<'a> {
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
        let mut fitter = FortranLeastSquaresQrDecomposition::new(self.data, None);
        let result = fitter.dqrls();
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

/*
This is a Rust implementation of the LINPACK dqrls subroutine. The original
Fortran code is in this file:
    * dqrls.f (/src/fortran/dqrls.f)[/src/bin/libdqrls.so]

Definition of the dqrls subroutine:
    * subroutine dqrls(x,n,p,y,ny,tol,b,rsd,qty,k,jpvt,qraux,work)
*/

extern "C" {
    /// This is the FFI function that calls the LINPACK dqrls subroutine.
    /// The function signature is:
    /// ```fortran
    /// subroutine dqrls(x,n,p,y,ny,tol,b,rsd,qty,k,jpvt,qraux,work)
    /// ```
    ///
    /// # Parameters
    /// * `x` is a matrix of real numbers representing the independent variables.
    /// * `n` is the number of rows in the matrix `x`.
    /// * `p` is the number of columns in the matrix `x`.
    /// * `y` is a matrix of real numbers representing the dependent variables.
    /// * `ny` is the number of columns in the matrix `y`.
    /// * `tol` is the tolerance for determining the rank of the matrix.
    /// * `b` is a matrix of real numbers representing the coefficients in a linear model.
    /// * `rsd` is a matrix of real numbers representing the residuals.
    /// * `qty` is a matrix of real numbers representing the Q-transposed times Y matrix.
    /// * `k` is the number of columns used in the solution.
    /// * `jpvt` is the pivot vector for the matrix `x`.
    /// * `qraux` is auxiliary information for the QR decomposition.
    /// * `work` is a work array.
    fn f_dqrls(
        x: *mut f64,      // Matrix X (modified in place)
        n: *const c_int,  // Number of rows in X
        p: *const c_int,  // Number of columns in X
        y: *const f64,    // Right-hand side matrix Y
        ny: *const c_int, // Number of columns in Y
        tol: *const f64,  // Tolerance for determining rank
        b: *mut f64,      // Solution matrix B
        rsd: *mut f64,    // Residual matrix
        qty: *mut f64,    // Q-transposed times Y matrix
        k: *mut c_int,    // Number of columns used in solution
        jpvt: *mut c_int, // Pivot vector for X
        qraux: *mut f64,  // Auxiliary information for QR decomposition
        work: *mut f64,   // Work array
    );
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

type ColumnPivot = Vec<c_int>;
type AuxiliaryInformation = Vec<f64>;
type Work = Vec<f64>;

#[derive(Debug, Builder)]
pub struct FortranLeastSquaresQrDecomposition<'a> {
    pub data: &'a Data,
    pub tol: Option<f64>,
}

impl<'a> FortranLeastSquaresQrDecomposition<'a> {
    pub fn new(data: &'a Data, tol: Option<f64>) -> Self {
        Self { data, tol }
    }

    pub fn dqrls(&mut self) -> Result<FortranLeastSquaresReturn, FortranLeastSquaresError> {
        let (n_rows, n_cols, n_cols_y) = self.get_dimensions();
        let (mut coefficients, mut residuals, mut qty) = self.allocate_solution_arrays();
        let (mut jpvt, mut qraux, mut work) = self.allocate_auxiliary_arrays();

        // Initialize number of columns used (k)
        let mut n_columns_used: i32 = 0 as c_int;

        unsafe {
            f_dqrls(
                self.data.x.as_slice().unwrap().as_ptr() as *mut f64,
                &n_rows,
                &n_cols,
                self.data.y.as_slice().unwrap().as_ptr(),
                &n_cols_y,
                &self.tol(),
                coefficients.as_slice_mut().unwrap().as_mut_ptr(),
                residuals.as_slice_mut().unwrap().as_mut_ptr(),
                qty.as_slice_mut().unwrap().as_mut_ptr(),
                &mut n_columns_used,
                jpvt.as_mut_ptr(),
                qraux.as_mut_ptr(),
                work.as_mut_ptr(),
            );
        }

        // Return the computed matrices
        Ok(FortranLeastSquaresReturn::builder()
            .beta(coefficients)
            .residuals(residuals)
            .q_transposed_times_y(qty)
            .qr_decomp_auxiliary_information(RealMatrix::from_vec(qraux, n_cols as usize, None))
            .build()
            .unwrap())
    }

    /// Get the dimensions of the input matrices
    ///
    /// # Example
    /// ```
    /// use lm::dqrls::{FortranLeastSquaresQrDecomposition, LinearSystem, RealMatrix};
    /// use ndarray::Array2;
    ///
    /// let x = RealMatrix::with_shape(3, 2);
    /// let y = RealMatrix::with_shape(3, 1);
    /// let dqrls = FortranLeastSquaresQrDecomposition::new(x, y, None);
    /// assert_eq!(dqrls.get_dimensions(), (3, 2, 1));
    /// assert_eq!(dqrls.get_dimensions(), (x.n_rows() as i32, x.n_cols() as i32, y.n_cols() as i32));
    /// ```
    pub fn get_dimensions(&self) -> (c_int, c_int, c_int) {
        (
            self.data.x().n_rows() as c_int,
            self.data.x().n_cols() as c_int,
            self.data.y().n_cols() as c_int,
        )
    }

    /// Allocate solution arrays for the QR decomposition
    ///
    /// # Example
    /// ```
    /// use lm::dqrls::{FortranLeastSquaresQrDecomposition, RealMatrix};
    /// use ndarray::Array2;
    ///
    /// let x = RealMatrix::with_shape(3, 2);
    /// let y = RealMatrix::with_shape(3, 1);
    /// let dqrls = FortranLeastSquaresQrDecomposition::new(x, y, None);
    /// let (b, rsd, qty) = dqrls.allocate_solution_arrays();
    /// assert_eq!(b, RealMatrix::with_shape(3, 1));
    /// assert_eq!(rsd, RealMatrix::with_shape(3, 1));
    /// assert_eq!(qty, RealMatrix::with_shape(3, 1));
    /// ```
    pub fn allocate_solution_arrays(&self) -> (RealMatrix, RealMatrix, RealMatrix) {
        (
            RealMatrix::with_shape(self.data.x().n_rows(), self.data.y().n_cols()), // Solution vector
            RealMatrix::with_shape(self.data.x().n_rows(), self.data.y().n_cols()), // Residuals
            RealMatrix::with_shape(self.data.x().n_rows(), self.data.y().n_cols()), // Q-transposed times Y
        )
    }

    /// Allocate auxiliary arrays for the QR decomposition
    ///
    /// # Example
    /// ```
    /// use lm::dqrls::{FortranLeastSquaresQrDecomposition, RealMatrix};
    ///
    /// let x = RealMatrix::with_shape(3, 2);
    /// let y = RealMatrix::with_shape(3, 1);
    /// let dqrls = FortranLeastSquaresQrDecomposition::new(x, y, None);
    /// let (jpvt, qraux, work) = dqrls.allocate_auxiliary_arrays();
    /// assert_eq!(jpvt, vec![0, 0]);
    /// assert_eq!(qraux, vec![0.0, 0.0]);
    /// assert_eq!(work, vec![0.0, 0.0, 0.0, 0.0]);
    /// ```
    pub fn allocate_auxiliary_arrays(&self) -> (ColumnPivot, AuxiliaryInformation, Work) {
        (
            vec![0 as c_int; self.data.x().n_cols()], // Column pivot vector
            vec![0.0; self.data.x().n_cols()],        // Auxiliary information
            vec![0.0; self.data.x().n_cols()],        // Work array
        )
    }
    /// Get the tolerance for determining the rank of the matrix
    ///
    /// # Example
    /// ```
    /// use lm::dqrls::{FortranLeastSquaresQrDecomposition, RealMatrix};
    ///
    /// let x = RealMatrix::with_shape(3, 2);
    /// let y = RealMatrix::with_shape(3, 1);
    /// let dqrls = FortranLeastSquaresQrDecomposition::new(x, y, Some(1e-10));
    /// assert_eq!(dqrls.tol(), 1e-10);
    /// ```
    fn tol(&self) -> f64 {
        self.tol.unwrap_or(1e-10)
    }
}

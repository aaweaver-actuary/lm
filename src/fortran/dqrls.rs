//! This module wraps the `dqrls` Fortran subroutine for solving the linear least squares problem.
//! The `dqrls` subroutine is a wrapper around the LAPACK `DGELS` routine, which uses the QR
//! decomposition method to solve the linear least squares problem.
//!
//! This module first defines the FFI function signature for the `dqrls` subroutine, and then
//! provides a safe interface for calling the subroutine from Rust.
//!
//! The `FortranDqrls` struct provides a safe interface for calling the `dqrls` subroutine from Rust.
//!
//! The original Fortran code for the `dqrls` subroutine can be found in the `dqrls.f` file in the
//! `src/fortran/src` directory.
//!
//! The `dqrls` subroutine has the following signature:
//! ```fortran
//! subroutine dqrls(x,n,p,y,ny,tol,b,rsd,qty,k,jpvt,qraux,work)
//! ```

use crate::errors::LmFitterError;
use crate::types::{c_int, Data, RealMatrix, Tolerance};

/// Define the `FortranDqrls` struct, which provides a safe interface for calling the `dqrls`
/// subroutine from Rust.
pub struct FortranDqrls<'a> {
    data: &'a Data,
    tolerance: Tolerance,
    coefficients: RealMatrix,
    residuals: RealMatrix,
    q_transpose_y: RealMatrix,
    pivot_vector_for_x: Vec<i32>,
    qr_auxiliary_information: Vec<f64>,
    work_array: Vec<f64>,
}

/// Implement the `Dqrls` struct.
impl<'a> FortranDqrls<'a> {
    /// Create a new `Dqrls` struct.
    pub fn new(data: &'a Data, tolerance: Tolerance, coefficients: Option<RealMatrix>) -> Self {
        FortranDqrls {
            data,
            tolerance,
            coefficients: coefficients
                .unwrap_or_else(|| RealMatrix::with_shape(data.x().n_cols(), data.y().n_cols())),
            residuals: RealMatrix::with_shape(data.x().n_rows(), data.y().n_cols()),
            q_transpose_y: RealMatrix::with_shape(data.x().n_cols(), data.y().n_cols()),
            pivot_vector_for_x: vec![0; data.x().n_cols()],
            qr_auxiliary_information: vec![0.0; data.x().n_cols()],
            work_array: vec![0.0; data.x().n_cols()],
        }
    }

    /// Return a reference to the x matrix from the data struct.
    pub fn x(&self) -> &RealMatrix {
        self.data.x()
    }

    /// Return a reference to the y matrix from the data struct.
    pub fn y(&self) -> &RealMatrix {
        self.data.y()
    }

    /// Return the unwrapped tolerance value. If no tolerance value is provided, the default value
    /// is used.
    pub fn tol(&self) -> f64 {
        self.tolerance.value()
    }

    /// Call the `dqrls` subroutine to solve the linear least squares problem.
    pub fn solve(&mut self) -> Result<&RealMatrix, LmFitterError> {
        unsafe {
            dqrls_(
                self.data.x.as_slice().unwrap().as_ptr() as *mut f64,
                self.x().n_rows() as *const c_int,
                self.x().n_cols() as *const c_int,
                self.data.y.as_slice().unwrap().as_ptr(),
                self.y().n_cols() as *const c_int,
                &self.tol() as *const f64,
                self.coefficients.values.as_mut_ptr(),
                self.residuals.values.as_mut_ptr(),
                self.q_transpose_y.values.as_mut_ptr(),
                self.y().n_cols() as *mut c_int,
                self.pivot_vector_for_x.as_mut_ptr(),
                self.qr_auxiliary_information.as_mut_ptr(),
                self.work_array.as_mut_ptr(),
            );
        }

        // Return the coefficients
        let coefficients = &self.coefficients;
        Ok(coefficients)
    }
}

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
    fn dqrls_(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{data, real_matrix::RealMatrix};

    #[test]
    fn test_dqrls_solve() {
        let x = RealMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, Some(2));
        let y = RealMatrix::from_vec(vec![1.0, 2.0, 3.0], 3, Some(1));
        let data = data::Data::new(x, y);
        let tolerance = Tolerance::default();
        let coefficients = RealMatrix::with_shape(2, 1);

        let mut dqrls = FortranDqrls::new(&data, tolerance, Some(coefficients));

        let result = dqrls.solve();

        assert!(result.is_ok());
    }
}

/*
/// Perform the QR decomposition using the LINPACK dqrls subroutine. This should be
/// the only place in the code where the LINPACK dqrls subroutine is called, and also
/// the only place where an unsafe block is needed.
///
/// # Arguments
/// * `x` - A pointer to the matrix of independent variables.
/// * `n` - A pointer to the number of rows in the matrix `x`.
/// * `p` - A pointer to the number of columns in the matrix `x`.
/// * `y` - A pointer to the matrix of dependent variables.
/// * `ny` - A pointer to the number of columns in the matrix `y`.
/// * `tol` - A pointer to the tolerance value for determining the rank of the matrix.
/// * `b` - A pointer to the solution vector.
/// * `rsd` - A pointer to the residuals.
/// * `qty` - A pointer to the Q-transposed times Y matrix.
/// * `k` - A pointer to the number of columns used in the solution.
/// * `jpvt` - A pointer to the pivot vector for the matrix `x`.
/// * `qraux` - A pointer to the auxiliary information for the QR decomposition.
///
/// # Returns
/// A vector of real numbers representing the solution vector.
///
/// # Safety
/// This function is unsafe because it calls an external FFI function that is not guaranteed
/// to be memory safe. The function signature is:
fn dqrls_subroutine(
    x: *mut f64,
    n: *const c_int,
    p: *const c_int,
    y: *const f64,
    ny: *const c_int,
    tol: *const f64,
    b: *mut f64,
    rsd: *mut f64,
    qty: *mut f64,
    k: *mut c_int,
    jpvt: *mut c_int,
    qraux: *mut f64,
    work: *mut f64,
) -> Result<Vec<f64>, LeastSquaresError> {
    unsafe {
        f_dqrls(
            x, n, p, y, ny, tol, b, rsd, qty, k, jpvt, qraux, work,
        );
    }


} */

/*     /// Perform the QR decomposition using the LINPACK dqrls subroutine.
   pub fn perform_qr_decomposition(
       &mut self,
   ) -> Result<FortranLeastSquaresReturn, LeastSquaresError> {
       let (mut jpvt, mut qraux, mut work) = self.allocate_auxiliary_arrays();

       // Initialize number of columns used (k)
       let mut n_columns_used: i32 = 0 as c_int;

       unsafe {
           f_dqrls(
               self.data.x.as_slice().unwrap().as_ptr() as *mut f64,
               self.x().n_rows() as *const c_int,
               self.x().n_cols() as *const c_int,
               self.data.y.as_slice().unwrap().as_ptr(),
               self.y().n_cols() as *const c_int,
               &self.tol(),
               // coefficients.as_slice_mut().unwrap().as_mut_ptr(),
               self.beta_array.as_mut_ptr(),
               // residuals.as_slice_mut().unwrap().as_mut_ptr(),
               self.residual_array.as_mut_ptr(),
               // qty.as_slice_mut().unwrap().as_mut_ptr(),
               self.q_transposed_times_y.as_mut_ptr(),
               &mut n_columns_used,
               jpvt.as_mut_ptr(),
               qraux.as_mut_ptr(),
               work.as_mut_ptr(),
           );
       }

       // Return the computed matrices
       Ok(FortranLeastSquaresReturn::builder()
           .beta(RealMatrix::from_vec(
               self.beta_array,
               self.x().n_rows(),
               Some(self.y().n_cols()),
           ))
           .residuals(RealMatrix::from_vec(
               self.residual_array,
               self.x().n_rows(),
               Some(self.y().n_cols()),
           ))
           .q_transposed_times_y(RealMatrix::from_vec(
               self.q_transposed_times_y,
               self.x().n_rows(),
               Some(self.y().n_cols()),
           ))
           .qr_decomp_auxiliary_information(RealMatrix::from_vec(qraux, n_cols as usize, None))
           .build()
           .unwrap())
   }
*/

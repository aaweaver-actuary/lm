// src/fitters/fit.rs

use super::qr_decomposition_fitter::QrDecompositionFitter;
use crate::errors::LmFitterError;
use crate::RealMatrix;

/// A trait for fitting a linear regression model to a dataset.
///
/// This trait is implemented for the `LinearSystemSolver` enum, which allows the user to choose
/// the desired strategy for solving the linear system. Each strategy is implemented in a separate
/// module, and must also implement the `SolveLinearRegression` trait.
pub trait FitLinearModel {
    /// Fit the linear regression model to the data.
    fn fit(&self) -> Result<RealMatrix, LmFitterError>;

    /// Get the x matrix.
    fn x(&self) -> &RealMatrix;

    /// Get the y matrix.
    fn y(&self) -> &RealMatrix;
}

/// An enum representing the available strategies for fitting a linear model to a dataset.
#[derive(Debug)]
pub enum LinearModelFitter<'a> {
    /// Fit the linear model using the QR decomposition method.
    QrDecomposition(QrDecompositionFitter<'a>),
}

impl<'a> FitLinearModel for LinearModelFitter<'a> {
    fn fit(&self) -> Result<RealMatrix, LmFitterError> {
        match self {
            LinearModelFitter::QrDecomposition(fitter) => fitter.fit(),
        }
    }

    fn x(&self) -> &RealMatrix {
        match self {
            LinearModelFitter::QrDecomposition(fitter) => fitter.x(),
        }
    }

    fn y(&self) -> &RealMatrix {
        match self {
            LinearModelFitter::QrDecomposition(fitter) => fitter.y(),
        }
    }
}

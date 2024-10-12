pub mod gram_schmidt_qr_factorizer;
pub mod qr_factorization;

use crate::types::RealMatrix;

/// A trait for factorizing a matrix using the QR method.
pub trait FactorizeQr {
    /// Compute the QR factorization of a matrix.
    fn qr(&self) -> (RealMatrix, RealMatrix);
}

/// An enum representing the available strategies for factorizing
/// a matrix using the QR method.
pub enum QrFactorizer {
    /// Factorize the matrix using the Householder reflection method.
    Householder(HouseholderQrFactorizer),

    /// Factorize the matrix using the Givens rotation method.
    Givens(GivensQrFactorizer),

    /// Factorize the matrix using the Gram-Schmidt method.
    GramSchmidt(GramSchmidtQrFactorizer),
}

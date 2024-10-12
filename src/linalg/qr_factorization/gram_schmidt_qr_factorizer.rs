use crate::types::RealMatrix;
use crate::linalg::qr_factorization::FactorizeQr;

/// A struct for factorizing a matrix using the Gram-Schmidt method.
#[derive(Debug, Clone)]
pub struct GramSchmidtQrFactorizer {
    matrix: &RealMatrix,
}

impl GramSchmidtQrFactorizer {
    /// Construct a new Gram-Schmidt factorizer.
    pub fn new(matrix: &RealMatrix) -> Self {
        GramSchmidtQrFactorizer { matrix }
    }
}

impl FactorizeQr for GramSchmidtQrFactorizer {
    fn qr(&self) -> (RealMatrix, RealMatrix) {
        todo!()
    }
}
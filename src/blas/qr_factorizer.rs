
use crate::types::RealMatrix;

pub trait FactorizeQr {
    fn qr(&self) -> (RealMatrix, RealMatrix);
}
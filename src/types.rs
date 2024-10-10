//! This module contains the types used in the library.

/// Re-export the `RealMatrix` type from the `real_matrix` module.
pub use crate::real_matrix::RealMatrix;

/// Re-export the `Data` type from the `data` module.
pub use crate::data::Data;

/// Re-export the C types used in the library.
pub use libc::{c_double, c_int};

/// An enum representing whether a real number is positive, negative, or zero.
#[derive(Debug, PartialEq)]
pub enum RealNumberSign {
    Positive(f64),
    Negative(f64),
    Zero,
}

/// The default tolerance value used in the library.
const DEFAULT_TOL: f64 = 1e-10;

/// A tuple struct that wraps a tolerance value for comparing floating point numbers
/// to determine whether or not a numerical method has converged.
///
/// * Note: On creation, the tolerance value validates that it is greater than zero.
#[derive(Debug, PartialEq, Clone)]
pub struct Tolerance(f64);

impl Tolerance {
    /// Create a new `Tolerance` struct.
    ///
    /// # Arguments
    /// * `value` - An optional `f64` value representing the tolerance value. If `None`, the default value of `1e-10` is used.
    ///
    /// # Panics
    /// Panics if the tolerance value is less than or equal to zero (since this doesn't make any sense).
    ///
    /// # Returns
    /// A new `Tolerance` struct, either with the default value or the user-specified value.
    ///
    /// # Examples
    /// ```
    /// use lm::types::{Tolerance, DEFAULT_TOL};
    /// const USER_SPECIFIED_TOL: f64 = 1e-5;
    ///
    /// let user_specified_tol = Tolerance::new(Some(USER_SPECIFIED_TOL));
    /// assert_eq!(user_specified_tol.value(), USER_SPECIFIED_TOL);
    ///
    /// let default_tol = Tolerance::new(None);
    /// assert_eq!(default_tol.value(), DEFAULT_TOL);
    ///
    /// // Panics if the tolerance value is less than or equal to zero.
    /// // Tolerance::new(Some(0.0));
    /// ```
    pub fn new(value: Option<f64>) -> Self {
        match value {
            Some(value) => {
                assert!(value > 0.0, "Tolerance value must be greater than zero.");
                Tolerance(value)
            }
            None => Tolerance(DEFAULT_TOL),
        }
    }

    /// Return the value of the tolerance. The actual value is stored in a private field, so this
    /// method is used to access the value.
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Implement the `Default` trait for the `Tolerance` struct. This allows the `Tolerance` struct to
/// be created with the default value of `1e-10` if no value is specified.
impl Default for Tolerance {
    fn default() -> Self {
        Tolerance(DEFAULT_TOL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const USER_SPECIFIED_TOL: f64 = 1e-5;

    #[test]
    fn test_tolerance_new() {
        let user_specified_tol = Tolerance::new(Some(USER_SPECIFIED_TOL));
        assert_eq!(user_specified_tol.value(), USER_SPECIFIED_TOL);

        let default_tol = Tolerance::new(None);
        assert_eq!(default_tol.value(), DEFAULT_TOL);
    }

    #[test]
    fn test_zero_tolerance_does_actually_panic() {
        let result = std::panic::catch_unwind(|| {
            Tolerance::new(Some(0.0));
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_negative_tolerance_does_actually_panic() {
        let result = std::panic::catch_unwind(|| {
            Tolerance::new(Some(-1.0));
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_tolerance_default() {
        let default_tol = Tolerance::default();
        assert_eq!(default_tol.value(), DEFAULT_TOL);
    }

    #[test]
    fn test_real_number_sign() {
        let positive = RealNumberSign::Positive(1.0);
        let negative = RealNumberSign::Negative(-1.0);
        let zero = RealNumberSign::Zero;

        assert_eq!(positive, RealNumberSign::Positive(1.0));
        assert_eq!(negative, RealNumberSign::Negative(-1.0));
        assert_eq!(zero, RealNumberSign::Zero);
    }
}

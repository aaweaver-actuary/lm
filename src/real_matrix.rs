// src/real_matrix.rs

use ndarray::Array2;

/// A struct representing a matrix of real numbers. The RealMatrix struct is a wrapper around
/// the ndarray::Array2 type, which is a two-dimensional array that is optimized for numerical
/// computations. The RealMatrix struct provides a more user-friendly interface for working with
/// the subset of ndarray's functionality that is needed for this project.
///
/// The RealMatrix struct is used to represent the x and y matrices in the linear regression model,
/// and is the primary data structure used to store the data for the model.
#[derive(Debug, Clone, PartialEq)]
pub struct RealMatrix {
    pub values: Array2<f64>,
}

impl Iterator for RealMatrix {
    type Item = f64;

    /// Implement the Iterator trait for the RealMatrix struct. This allows the user to iterate
    /// over the elements of the matrix using a for loop or other iterator methods.
    /// The iterator is implemented as a simple wrapper around the iterator for the underlying
    /// ndarray::Array2 type.
    fn next(&mut self) -> Option<Self::Item> {
        self.values.iter().next().copied()
    }
}

impl RealMatrix {
    /// Create a new RealMatrix instance from an ndarray::Array2.
    pub fn new(data: Array2<f64>) -> Self {
        RealMatrix { values: data }
    }

    /// Create a new RealMatrix instance with the specified number of rows and columns.
    /// The matrix is initialized with zeros.
    ///
    /// # Arguments
    /// * `n_rows` - The number of rows in the matrix.
    /// * `n_cols` - The number of columns in the matrix.
    ///
    /// # Returns
    /// A new RealMatrix instance with the specified shape, initialized with zeros.
    ///
    /// # Example
    /// ```
    /// use lm::types::RealMatrix;
    ///
    /// let matrix = RealMatrix::with_shape(3, 2);
    /// assert_eq!(matrix.n_rows(), 3);
    /// assert_eq!(matrix.n_cols(), 2);
    ///
    /// // The matrix is initialized with zeros
    /// for value in matrix.values.iter() {
    ///    assert_eq!(*value, 0.0);
    /// }
    ///
    /// // The matrix is not empty
    /// assert!(!matrix.is_empty());
    /// ```
    pub fn with_shape(n_rows: usize, n_cols: usize) -> Self {
        RealMatrix {
            values: Array2::<f64>::zeros((n_rows, n_cols)),
        }
    }

    pub fn dot(&self, other: &RealMatrix) -> RealMatrix {
        RealMatrix {
            values: self.values.dot(&other.values),
        }
    }

    pub fn transpose(&self) -> RealMatrix {
        RealMatrix {
            values: self.values.to_owned().reversed_axes(),
        }
    }

    pub fn plus(&self, other: &RealMatrix) -> RealMatrix {
        RealMatrix {
            values: &self.values + &other.values,
        }
    }

    pub fn minus(&self, other: &RealMatrix) -> RealMatrix {
        RealMatrix {
            values: &self.values - &other.values,
        }
    }

    pub fn from_vec(data: Vec<f64>, n_rows: usize, n_cols: Option<usize>) -> Self {
        RealMatrix {
            values: Array2::<f64>::from_shape_vec((n_rows, n_cols.unwrap_or(1)), data).unwrap(),
        }
    }

    pub fn as_slice(&self) -> Option<&[f64]> {
        self.values.as_slice()
    }

    pub fn as_slice_mut(&mut self) -> Option<&mut [f64]> {
        self.values.as_slice_mut()
    }

    pub fn shape(&self) -> &[usize; 2] {
        self.values
            .shape()
            .try_into()
            .expect("Shape should have exactly 2 elements")
    }

    pub fn n_rows(&self) -> usize {
        self.values.shape()[0]
    }

    pub fn n_cols(&self) -> usize {
        self.values.shape()[1]
    }

    pub fn is_vec(&self) -> bool {
        self.n_cols() == 1 || self.n_rows() == 1
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn ndim(&self) -> usize {
        self.values.ndim()
    }
}

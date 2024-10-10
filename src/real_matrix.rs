use ndarray::Array2;

#[derive(Debug, Clone, PartialEq)]
pub struct RealMatrix {
    pub values: Array2<f64>,
}

impl RealMatrix {
    pub fn new(data: Array2<f64>) -> Self {
        RealMatrix { values: data }
    }

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

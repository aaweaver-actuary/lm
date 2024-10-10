// src/linear_model.rs

use crate::fitters::fit::FitModel;
use crate::{Data, RealMatrix};
use std::cmp::Ordering::{Equal, Greater, Less};

#[derive(Debug, PartialEq)]
pub enum LinearModel<'a> {
    Fitted(FittedLinearModel<'a>),
    Unfitted(UnfittedLinearModel<'a>),
}

impl<'a> PartialOrd for LinearModel<'a> {
    /// Compare two `LinearModel` instances. This is mainly useful for testing purposes. Fitted > Unfitted.
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (LinearModel::Fitted(_), LinearModel::Unfitted(_)) => Some(Greater),
            (LinearModel::Unfitted(_), LinearModel::Fitted(_)) => Some(Less),
            (LinearModel::Fitted(_), LinearModel::Fitted(_)) => Some(Equal),
            (LinearModel::Unfitted(_), LinearModel::Unfitted(_)) => Some(Equal),
        }
    }
}

impl<'a> LinearModel<'a> {
    pub fn new(data: &'a Data) -> Self {
        LinearModel::Unfitted(UnfittedLinearModel::new(data))
    }

    pub fn update_coefficients(&mut self, coefficients: RealMatrix) {
        match self {
            LinearModel::Fitted(fitted) => fitted.update_coefficients(coefficients),
            LinearModel::Unfitted(_) => (),
        }
    }

    pub fn data(&self) -> &Data {
        match self {
            LinearModel::Fitted(fitted) => fitted.data,
            LinearModel::Unfitted(unfitted) => unfitted.data,
        }
    }

    pub fn x(&self) -> &RealMatrix {
        match self {
            LinearModel::Fitted(fitted) => &fitted.data.x(),
            LinearModel::Unfitted(unfitted) => &unfitted.data.x(),
        }
    }

    pub fn y(&self) -> &RealMatrix {
        match self {
            LinearModel::Fitted(fitted) => &fitted.data.y(),
            LinearModel::Unfitted(unfitted) => &unfitted.data.y(),
        }
    }

    pub fn fit(&mut self, fitter: &impl FitModel) {
        match self {
            // If already fitted, re-fit the model.
            LinearModel::Fitted(_) => {
                self.update_coefficients(fitter.fit().unwrap());
            }

            // If unfitted, fit the model and update the enum variant.
            LinearModel::Unfitted(unfitted_model) => {
                *self = LinearModel::Fitted(FittedLinearModel {
                    data: unfitted_model.data,
                    coefficients: fitter.fit().unwrap(),
                });
            }
        }
    }

    pub fn coefficients(&self) -> Option<&RealMatrix> {
        match self {
            LinearModel::Fitted(fitted) => Some(&fitted.coefficients),
            LinearModel::Unfitted(_) => None,
        }
    }

    pub fn residuals(&self) -> Option<RealMatrix> {
        match self {
            LinearModel::Fitted(fitted) => Some(fitted.residuals()),
            LinearModel::Unfitted(_) => None,
        }
    }

    pub fn predict(&self, x: Option<&RealMatrix>) -> Option<RealMatrix> {
        match self {
            LinearModel::Fitted(fitted) => Some(fitted.predict(x)),
            LinearModel::Unfitted(_) => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct FittedLinearModel<'a> {
    pub data: &'a Data,
    pub coefficients: RealMatrix,
}

impl<'a> FittedLinearModel<'a> {
    pub fn new(data: &'a Data, coefficients: RealMatrix) -> Self {
        FittedLinearModel { data, coefficients }
    }

    pub fn update_coefficients(&mut self, coefficients: RealMatrix) {
        self.coefficients = coefficients;
    }

    pub fn predict(&self, x: Option<&RealMatrix>) -> RealMatrix {
        match x {
            // If x is provided, use it to make predictions.
            Some(x) => x.dot(&self.coefficients),

            // If x is not provided, use the data's x matrix to make predictions.
            None => self.data.x().dot(&self.coefficients),
        }
    }

    pub fn residuals(&self) -> RealMatrix {
        self.data.y().minus(&self.predict(Some(self.data.x())))
    }
}

#[derive(Debug, PartialEq)]
pub struct UnfittedLinearModel<'a> {
    pub data: &'a Data,
}

impl<'a> UnfittedLinearModel<'a> {
    pub fn new(data: &'a Data) -> Self {
        UnfittedLinearModel { data }
    }
}

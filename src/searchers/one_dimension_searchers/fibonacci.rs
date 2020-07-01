use super::search::OneDimensionalSearcher;
use crate::searchers::extremum_searcher;
use crate::searchers::extremum_searcher::IterationResult;

use nalgebra::RealField;
use std::sync::Arc;

#[derive(Clone)]
pub struct Fibonacci<Scalar> {
    f: Arc<dyn Fn(Scalar) -> Scalar>,
    left: Scalar,
    right: Scalar,
    fn1: u128,
    fn2: u128,
    x1: Scalar,
    x2: Scalar,
    f1: Scalar,
    f2: Scalar,
    dx: Scalar,
    iters: usize,
    func_calls: usize,
    eps: Scalar,
    max_iters: usize,
    comparator: std::cmp::Ordering,
}

impl<Scalar> Fibonacci<Scalar>
where
    Scalar: RealField,
{
    pub fn new(
        left: Scalar,
        right: Scalar,
        f: Arc<dyn Fn(Scalar) -> Scalar>,
        comparator: std::cmp::Ordering,
        eps: Scalar,
        max_iters: usize,
    ) -> Self {
        let mut fn1 = 1u128;
        let mut fn2 = 1u128;
        while Scalar::from_u128(fn2).unwrap() <= (right - left) / eps {
            let ft = fn2;
            fn2 += fn1;
            fn1 = ft;
        }
        let x1 = left
            + Scalar::from_u128(fn2 - fn1).unwrap() / Scalar::from_u128(fn2).unwrap()
                * (right - left);
        let x2 = left
            + Scalar::from_u128(fn1).unwrap() / Scalar::from_u128(fn2).unwrap() * (right - left);
        let f1 = f(x1);
        let f2 = f(x2);
        Self {
            left,
            right,
            f,
            fn1,
            fn2,
            f1,
            f2,
            x1,
            x2,
            dx: Scalar::max_value(),
            iters: 0,
            func_calls: 2,
            eps,
            max_iters,
            comparator,
        }
    }
}

impl<Scalar> OneDimensionalSearcher<Scalar> for Fibonacci<Scalar> where Scalar: RealField {}

impl<Scalar> extremum_searcher::Search<Scalar> for Fibonacci<Scalar>
where
    Scalar: RealField,
{
    fn iters(&self) -> usize {
        self.iters
    }
    fn func_calls(&self) -> usize {
        self.func_calls
    }
    fn x(&self) -> Scalar {
        (self.right + self.left) / Scalar::from_i8(2).unwrap()
    }
    fn dx(&self) -> Scalar {
        self.dx
    }
}

impl<Scalar> Iterator for Fibonacci<Scalar>
where
    Scalar: RealField,
{
    type Item = IterationResult<Scalar>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.fn1 == 0 {
            return None;
        }
        let is_extra = if self.dx.abs() < self.eps || self.iters >= self.max_iters {
            true
        } else {
            false
        };
        let _x = (self.right + self.left) / Scalar::from_i8(2).unwrap();
        if self.f1.partial_cmp(&self.f2).unwrap() == self.comparator {
            self.right = self.x2;
            self.x2 = self.x1;
            self.x1 = self.left
                + Scalar::from_u128(self.fn2 - self.fn1).unwrap()
                    / Scalar::from_u128(self.fn2).unwrap()
                    * (self.right - self.left);
            self.f2 = self.f1;
            self.f1 = (self.f)(self.x1);
        } else {
            self.left = self.x1;
            self.x1 = self.x2;
            self.x2 = self.left
                + Scalar::from_u128(self.fn1).unwrap() / Scalar::from_u128(self.fn2).unwrap()
                    * (self.right - self.left);
            self.f1 = self.f2;
            self.f2 = (self.f)(self.x2);
        }
        let ft = self.fn1;
        self.fn1 = self.fn2 - self.fn1;
        self.fn2 = ft;
        let x = (self.right + self.left) / Scalar::from_i8(2).unwrap();
        let _dx = self.dx;
        self.dx = x - _x;
        self.iters += 1;
        self.func_calls += 1;
        Some(IterationResult::new(x, x - _x, 1, is_extra))
    }
}

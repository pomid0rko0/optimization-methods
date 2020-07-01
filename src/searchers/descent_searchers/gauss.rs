use super::search::DescentSearcher;
use crate::searchers::extremum_searcher;
use crate::searchers::extremum_searcher::IterationResult;
use crate::searchers::one_dimension_searchers;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, RealField, VectorN};
use std::sync::Arc;

pub struct Gauss<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    x: VectorN<Scalar, Dimension>,
    dx: VectorN<Scalar, Dimension>,
    func_calls: usize,
    iters: usize,
    f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
    eps: Scalar,
    max_iters: usize,
    comparator: std::cmp::Ordering,
}

impl<Scalar, Dimension> Gauss<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    pub fn new(
        x0: VectorN<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        comparator: std::cmp::Ordering,
        eps: Scalar,
        max_iters: usize,
    ) -> Self {
        Self {
            x: x0.clone(),
            dx: VectorN::<Scalar, Dimension>::from_element(Scalar::max_value()),
            f,
            eps,
            iters: 0,
            func_calls: 0,
            max_iters,
            comparator,
        }
    }
}

impl<Scalar, Dimension> DescentSearcher<Scalar, Dimension> for Gauss<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    fn S(&self) -> VectorN<Scalar, Dimension> {
        let mut s = nalgebra::zero::<VectorN<Scalar, Dimension>>();
        s[self.iters % Dimension::dim()] = Scalar::one();
        s
    }
}

impl<Scalar, Dimension> extremum_searcher::Search<VectorN<Scalar, Dimension>>
    for Gauss<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    fn iters(&self) -> usize {
        self.iters
    }
    fn func_calls(&self) -> usize {
        self.func_calls
    }
    fn x(&self) -> VectorN<Scalar, Dimension> {
        self.x.clone()
    }
    fn dx(&self) -> VectorN<Scalar, Dimension> {
        self.dx.clone()
    }
}

impl<Scalar, Dimension> Iterator for Gauss<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    type Item = IterationResult<VectorN<Scalar, Dimension>>;
    fn next(&mut self) -> Option<Self::Item> {
        let is_extra =
            if self.dx.iter().all(|xi| xi.abs() < self.eps) || self.iters >= self.max_iters {
                true
            } else {
                false
            };
        let x = self.x.clone();
        let f = self.f.clone();
        let s = self.S();
        let lambda_result = one_dimension_searchers::search::Search::result(
            Scalar::zero(),
            Arc::new(move |lambda| f(x.clone() + s.clone() * lambda)),
            self.comparator,
            one_dimension_searchers::search::Method::Fibonacci,
            self.eps,
            self.max_iters,
        );
        self.dx = self.S() * lambda_result.x();
        self.x += self.dx.clone();
        self.iters += 1;
        self.func_calls += lambda_result.func_calls();
        Some(IterationResult::new(
            self.x.clone(),
            self.dx.clone(),
            lambda_result.func_calls(),
            is_extra,
        ))
    }
}

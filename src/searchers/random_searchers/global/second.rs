use super::super::search::{Area, RandomSearcher};
use crate::searchers::descent_searchers;
use crate::searchers::descent_searchers::search::Method;
use crate::searchers::extremum_searcher;
use crate::searchers::extremum_searcher::IterationResult;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, RealField, VectorN};
use std::sync::Arc;

struct Second<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension> + Allocator<(Scalar, Scalar), Dimension>,
{
    f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
    comparator: std::cmp::Ordering,
    x: VectorN<Scalar, Dimension>,
    D: Area<Scalar, Dimension>,
    func_calls: usize,
    iters: usize,
    _f: Scalar,
    max_iters: usize,
    method: Method,
    eps: Scalar,
    got_result: bool,
}
impl<Scalar, Dimension> Second<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension> + Allocator<(Scalar, Scalar), Dimension>,
{
    pub fn new(
        mut D: Area<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        comparator: std::cmp::Ordering,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> Self {
        let result = descent_searchers::search::Search::result(
            D.get_random_point(),
            f.clone(),
            comparator,
            method.clone(),
            eps,
            max_iters,
        );
        let x = result.x();
        let _f = f(x.clone());
        Self {
            comparator,
            x,
            D,
            f,
            _f,
            iters: 1,
            func_calls: result.func_calls(),
            max_iters,
            method,
            eps,
            got_result: false,
        }
    }
}
impl<Scalar, Dimension> RandomSearcher<Scalar, Dimension> for Second<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension> + Allocator<(Scalar, Scalar), Dimension>,
{
}
impl<Scalar, Dimension> extremum_searcher::Search<VectorN<Scalar, Dimension>>
    for Second<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension> + Allocator<(Scalar, Scalar), Dimension>,
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
        self.x.clone()
    }
}
impl<Scalar, Dimension> Iterator for Second<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension> + Allocator<(Scalar, Scalar), Dimension>,
{
    type Item = IterationResult<VectorN<Scalar, Dimension>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iters += 1;
        for _ in 0..self.max_iters {
            let result = super::super::search::Search::result(
                self.D,
                self.f.clone(),
                self.comparator,
                super::super::search::Method::Simple,
                self.eps,
                self.eps,
                self.max_iters,
            );
            let x = result.x();
            let dx = self.x.clone() - x.clone();
            let f = (self.f)(x.clone());
            self.func_calls += result.func_calls() + 1;
            if self._f.partial_cmp(&f).unwrap() == self.comparator {
                self._f = f;
                self.x = x.clone();
                return Some(IterationResult::new(x, dx, 1, self.got_result));
            }
        }
        self.got_result = true;
        Some(IterationResult::new(
            self.x.clone(),
            VectorN::<Scalar, Dimension>::from_element(Scalar::zero()),
            1,
            self.got_result,
        ))
    }
}

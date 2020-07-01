use super::search::{Area, RandomSearcher};
use crate::searchers::extremum_searcher;
use crate::searchers::extremum_searcher::IterationResult;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, RealField, VectorN};
use std::sync::Arc;

pub struct Simple<Scalar, Dimension>
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
}

impl<Scalar, Dimension> Simple<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension> + Allocator<(Scalar, Scalar), Dimension>,
{
    pub fn new(
        x0: VectorN<Scalar, Dimension>,
        f0: Scalar,
        mut D: Area<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        comparator: std::cmp::Ordering,
        eps: VectorN<Scalar, Dimension>,
        alpha: Scalar,
    ) -> Self {
        let x = D.get_random_point();
        let _f = f(x.clone());
        let v = D.V();
        Self {
            comparator,
            x,
            D,
            f,
            _f,
            iters: 1,
            func_calls: 1,
            max_iters: ((Scalar::one() - alpha).ln()
                / (Scalar::one() - eps.iter().fold(Scalar::one(), |result, &ei| result * ei) / v)
                    .ln())
            .ceil()
            .to_subset()
            .unwrap() as usize,
        }
    }
}

impl<Scalar, Dimension> RandomSearcher<Scalar, Dimension> for Simple<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension> + Allocator<(Scalar, Scalar), Dimension>,
{
}

impl<Scalar, Dimension> extremum_searcher::Search<VectorN<Scalar, Dimension>>
    for Simple<Scalar, Dimension>
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

impl<Scalar, Dimension> Iterator for Simple<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension> + Allocator<(Scalar, Scalar), Dimension>,
{
    type Item = IterationResult<VectorN<Scalar, Dimension>>;
    fn next(&mut self) -> Option<Self::Item> {
        let is_extra = self.iters >= self.max_iters;
        let x = self.D.get_random_point();
        let dx = self.x.clone() - x.clone();
        let f = (self.f)(x.clone());
        if self._f.partial_cmp(&f).unwrap() == self.comparator {
            self._f = f;
            self.x = x.clone();
        }
        self.func_calls += 1;
        self.iters += 1;
        Some(IterationResult::new(x, dx, 1, is_extra))
    }
}

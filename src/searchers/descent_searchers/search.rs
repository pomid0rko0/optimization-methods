use crate::searchers::extremum_searcher;
use crate::searchers::extremum_searcher::{FinalResult, IterationResult};
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, RealField, VectorN};
use std::sync::Arc;

pub trait DescentSearcher<Scalar, Dimension>:
    Iterator<Item = IterationResult<VectorN<Scalar, Dimension>>>
    + extremum_searcher::Search<VectorN<Scalar, Dimension>>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    #[allow(non_snake_case)]
    fn S(&self) -> VectorN<Scalar, Dimension>;
}

#[derive(Clone)]
pub enum Method {
    Gauss,
}

pub struct Search<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    x: VectorN<Scalar, Dimension>,
    dx: VectorN<Scalar, Dimension>,
    func_calls: usize,
    iters: usize,
    method: Box<
        dyn DescentSearcher<Scalar, Dimension, Item = IterationResult<VectorN<Scalar, Dimension>>>,
    >,
}

impl<Scalar, Dimension> Search<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    pub fn new(
        x0: VectorN<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        comparator: std::cmp::Ordering,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> Self {
        let m = match method {
            Method::Gauss => super::gauss::Gauss::new(x0.clone(), f, comparator, eps, max_iters),
        };
        use crate::searchers::extremum_searcher::Search;
        Self {
            x: x0,
            dx: VectorN::<Scalar, Dimension>::from_element(Scalar::max_value()),
            func_calls: m.func_calls(),
            iters: 0,
            method: Box::new(m),
        }
    }
    pub fn result(
        x0: VectorN<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        comparator: std::cmp::Ordering,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        use crate::searchers::extremum_searcher::Search;
        Self::new(x0, f, comparator, method, eps, max_iters).result()
    }
    pub fn Mnimimum(
        x0: VectorN<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        use crate::searchers::extremum_searcher::Search;
        Self::new(x0, f, std::cmp::Ordering::Less, method, eps, max_iters).result()
    }
    pub fn Maximum(
        x0: VectorN<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        use crate::searchers::extremum_searcher::Search;
        Self::new(x0, f, std::cmp::Ordering::Greater, method, eps, max_iters).result()
    }
}

impl<Scalar, Dimension> extremum_searcher::Search<VectorN<Scalar, Dimension>>
    for Search<Scalar, Dimension>
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

impl<Scalar, Dimension> Iterator for Search<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    type Item = IterationResult<VectorN<Scalar, Dimension>>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.method.next() {
            Some(r) => {
                self.iters += 1;
                self.func_calls += r.func_calls();
                self.x = r.x();
                self.dx = r.dx();
                Some(IterationResult::new(
                    self.x.clone(),
                    self.dx.clone(),
                    r.func_calls(),
                    r.is_extra(),
                ))
            }
            None => None,
        }
    }
}

use super::super::extremum_searcher;
use super::super::extremum_searcher::{FinalResult, IterationResult};
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, RealField, VectorN};
use rand::Rng;
use std::sync::Arc;

pub struct Area<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<(Scalar, Scalar), Dimension>,
{
    bounds: VectorN<(Scalar, Scalar), Dimension>,
    rng: rand::rngs::ThreadRng,
}

impl<Scalar, Dimension> Area<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<(Scalar, Scalar), Dimension> + Allocator<Scalar, Dimension>,
{
    pub fn new(bounds: VectorN<(Scalar, Scalar), Dimension>) -> Self {
        Self {
            bounds,
            rng: rand::thread_rng(),
        }
    }
    pub fn V(&self) -> Scalar {
        self.bounds
            .iter()
            .fold(Scalar::one(), |v, &(left, right)| v * (right - left))
    }
    pub fn get_random_point(&mut self) -> VectorN<Scalar, Dimension> {
        VectorN::<Scalar, Dimension>::from_vec(
            self.bounds
                .clone()
                .iter()
                .map(|&(left, right)| {
                    Scalar::from_subset(
                        &self
                            .rng
                            .gen_range(left.to_subset().unwrap(), right.to_subset().unwrap()),
                    )
                })
                .collect(),
        )
    }
}

pub trait RandomSearcher<Scalar, Dimension>:
    Iterator<Item = IterationResult<VectorN<Scalar, Dimension>>>
    + extremum_searcher::Search<VectorN<Scalar, Dimension>>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension> + Allocator<(Scalar, Scalar), Dimension>,
{
}

#[derive(Clone)]
pub enum Method {
    Simple,
    GlobalFirst,
    GlobalSecond,
    GlobalThird,
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
        dyn RandomSearcher<Scalar, Dimension, Item = IterationResult<VectorN<Scalar, Dimension>>>,
    >,
}

impl<Scalar, Dimension> Search<Scalar, Dimension>
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
        alpha: Scalar,
        max_iters: usize,
    ) -> Self {
        let m = match method {
            Method::Simple => super::simple::Simple::new(
                D,
                f,
                comparator,
                VectorN::<Scalar, Dimension>::from_element(eps),
                alpha,
            ),
            _ => todo! {},
        };
        use crate::searchers::extremum_searcher::Search;
        Self {
            x: m.x(),
            dx: m.dx(),
            func_calls: m.func_calls(),
            iters: 0,
            method: Box::new(m),
        }
    }
    pub fn result(
        mut D: Area<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        comparator: std::cmp::Ordering,
        method: Method,
        eps: Scalar,
        alpha: Scalar,
        max_iters: usize,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        use crate::searchers::extremum_searcher::Search;
        Self::new(D, f, comparator, method, eps, alpha, max_iters).result()
    }
    pub fn Mnimimum(
        mut D: Area<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        method: Method,
        eps: Scalar,
        alpha: Scalar,
        max_iters: usize,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        use crate::searchers::extremum_searcher::Search;
        Self::new(
            D,
            f,
            std::cmp::Ordering::Less,
            method,
            eps,
            alpha,
            max_iters,
        )
        .result()
    }
    pub fn Maximum(
        mut D: Area<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        method: Method,
        eps: Scalar,
        alpha: Scalar,
        max_iters: usize,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        use crate::searchers::extremum_searcher::Search;
        Self::new(
            D,
            f,
            std::cmp::Ordering::Greater,
            method,
            eps,
            alpha,
            max_iters,
        )
        .result()
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

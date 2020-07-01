extern crate rand;

use super::minimize;
use super::minimize::{FinalResult, IterationResult};
use super::one_dimension_searchers;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, RealField, VectorN};
use std::sync::Arc;

#[derive(Copy)]
pub struct Area<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
{
    bounds: [(Scalar, Scalar); Dimension::dim()],
}

impl<Scalar, Dimension> Area<Scalar, Dimension> {
    pub fn V(&self) -> Scalar {
        self.bounds
            .iter()
            .fold(Scalar::one(), |v, (left, right)| v * (right - left));
    }
    pub fn get_random_point(&self) -> VectorN<Scalar, Dimension> {
        VectorN::<Scalr, Dimension>::from(
            self.bounds
                .iter()
                .map(|(left, right)| self.rng.gen_range(left, right))
                .collect(),
        )
    }
}

trait RandomSearchMethod<Scalar, Dimension>:
    Iterator<Item = IterationResult<VectorN<Scalar, Dimension>>>
    + minimize::Minimize<VectorN<Scalar, Dimension>>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
}

struct Simple<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
    comparator: std::cmp::Ordering,
    x: VectorN<Scalar, Dimension>,
    D: Area<Scalar>,
    func_calls: usize,
    iters: usize,
    _f: Scalar,
    max_iters: usize,
    rgn: rand::rngs::ThreadRng,
}

impl<Scalar, Dimension> Simple<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    fn new(
        D: Area<Scalar>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        comparator: std::cmp::Ordering,
        eps: VectorN<Scalar, DImnesnion>,
        alpha: Scalar,
    ) -> Self {
        Self {
            comparator,
            x: D.get_random_point(),
            D,
            dx: VectorN::<Scalar, Dimension>::from_element(Scalar::max_value()),
            f,
            _f: f(x),
            iters: 1,
            func_calls: 1,
            max_iters: 1f64
                .log(
                    1 - alpha,
                    1 - eps.iter().fold(Scalar::one() | result, ei | result * ei) / D.V(),
                )
                .ceil(),
            rgn: rand::thread_rng(),
        }
    }
    pub fn result(
        D: Area<Scalar>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        comparator: std::cmp::Ordering,
        eps: VectorN<Scalar, DImnesnion>,
        alpha: Scalar,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        use crate::solvers::extremum_searcher::Search;
        Self::new(D, F, comparator, eps, alpha).result()
    }
    pub fn Mnimimum(
        D: Area<Scalar>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        comparator: std::cmp::Ordering,
        eps: VectorN<Scalar, DImnesnion>,
        alpha: Scalar,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        use crate::solvers::extremum_searcher::Search;
        Self::new(D, F, std::cmp::Ordering::Less, eps, alpha).result()
    }
    pub fn Maximum(
        D: Area<Scalar>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        comparator: std::cmp::Ordering,
        eps: VectorN<Scalar, DImnesnion>,
        alpha: Scalar,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        use crate::solvers::extremum_searcher::Search;
        Self::new(D, F, std::cmp::Ordering::Greater, eps, alpha).result()
    }
}

impl<Scalar, Dimension> RandomSearchMethod<Scalar, Dimension> for Simple<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
}

impl<Scalar, Dimension> minimize::Minimize<VectorN<Scalar, Dimension>> for Simple<Scalar, Dimension>
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

impl<Scalar, Dimension> Iterator for Simple<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    type Item = IterationResult<VectorN<Scalar, Dimension>>;
    fn next(&mut self) -> Option<Self::Item> {
        let is_extra = self.iters >= self.max_iters;
        let x = self.D.get_random_point();
        let dx = self.x - x;
        let f = (self.f)(x);
        if (self.f)(x).partial_cmp(f).unwrap() == self.comparator {
            self._f = f;
            self.x = x;
        }
        self.func_calls += 1;
        self.iters += 1;
        Some(IterationResult::new(x, dx))
    }
}

#[derive(Clone)]
pub enum Method {
    Simple,
}

pub struct Minimize<Scalar, Dimension>
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
        dyn DescentMethod<Scalar, Dimension, Item = IterationResult<VectorN<Scalar, Dimension>>>,
    >,
}

impl<Scalar, Dimension> Minimize<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    pub fn new(
        x0: VectorN<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> Self {
        let m = match method {
            Method::Gauss => Gauss::new(x0.clone(), f, eps, max_iters),
        };
        Self {
            x: x0,
            dx: VectorN::<Scalar, Dimension>::from_element(Scalar::max_value()),
            func_calls: m.func_calls,
            iters: 0,
            method: Box::new(m),
        }
    }
    pub fn result(
        x0: VectorN<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        <Self as minimize::Minimize<VectorN<Scalar, Dimension>>>::result(&mut Self::new(
            x0, f, method, eps, max_iters,
        ))
    }
}

impl<Scalar, Dimension> minimize::Minimize<VectorN<Scalar, Dimension>>
    for Minimize<Scalar, Dimension>
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

impl<Scalar, Dimension> Iterator for Minimize<Scalar, Dimension>
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

use super::minimize;
use super::minimize::{FinalResult, IterationResult};
use super::one_dimension_searchers;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, RealField, VectorN};
use std::sync::Arc;

trait DescentMethod<Scalar, Dimension>:
    Iterator<Item = IterationResult<VectorN<Scalar, Dimension>>>
    + minimize::Minimize<VectorN<Scalar, Dimension>>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    #[allow(non_snake_case)]
    fn S(&self) -> VectorN<Scalar, Dimension>;
}

struct Gauss<Scalar, Dimension>
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
}

impl<Scalar, Dimension> Gauss<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    fn new(
        x0: VectorN<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
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
        }
    }
}

impl<Scalar, Dimension> DescentMethod<Scalar, Dimension> for Gauss<Scalar, Dimension>
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

impl<Scalar, Dimension> minimize::Minimize<VectorN<Scalar, Dimension>> for Gauss<Scalar, Dimension>
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
        let lambda_result = one_dimension_searchers::Minimize::result(
            Scalar::zero(),
            Arc::new(move |lambda| f(x.clone() + s.clone() * lambda)),
            one_dimension_searchers::Method::Fibonacci,
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

#[derive(Clone)]
pub enum Method {
    Gauss,
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

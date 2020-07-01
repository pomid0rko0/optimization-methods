use super::descent_methods;
use super::descent_methods::Method;
use super::extremum_searcher;
use super::extremum_searcher::{FinalResult, IterationResult};
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, RealField, VectorN};
use std::sync::Arc;

#[derive(Clone)]
pub enum BoundType {
    Equal,
    Unequal,
}

#[derive(Clone)]
pub struct Bound<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    function: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
    bound_type: BoundType,
    penalty: Arc<dyn Fn(Scalar) -> Scalar>,
    coefficient: Scalar,
    coefficient_function: Arc<dyn Fn(Scalar) -> Scalar>,
}

impl<Scalar, Dimension> Bound<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    pub fn new(
        function: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        bound_type: BoundType,
        penalty: Arc<dyn Fn(Scalar) -> Scalar>,
        coefficient: Scalar,
        coefficient_function: Arc<dyn Fn(Scalar) -> Scalar>,
    ) -> Self {
        Self {
            function,
            bound_type,
            penalty,
            coefficient,
            coefficient_function,
        }
    }
}

pub struct Search<Scalar, Dimension>
where
    Scalar: RealField,
    Dimension: DimName,
    DefaultAllocator: Allocator<Scalar, Dimension>,
{
    x: VectorN<Scalar, Dimension>,
    dx: VectorN<Scalar, Dimension>,
    f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
    comparator: std::cmp::Ordering,
    func_calls: usize,
    iters: usize,
    method: Method,
    max_iters: usize,
    eps: Scalar,
    g: Vec<Bound<Scalar, Dimension>>,
    got_result: bool,
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
        g: Vec<Bound<Scalar, Dimension>>,
        eps: Scalar,
        max_iters: usize,
    ) -> Self {
        Self {
            x: x0,
            f,
            comparator,
            dx: VectorN::<Scalar, Dimension>::from_element(Scalar::max_value()),
            iters: 0,
            func_calls: 0,
            method,
            max_iters,
            g,
            eps,
            got_result: false,
        }
    }
    pub fn result(
        x0: VectorN<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        comparator: std::cmp::Ordering,
        method: Method,
        g: Vec<Bound<Scalar, Dimension>>,
        eps: Scalar,
        max_iters: usize,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        use crate::searchers::extremum_searcher::Search;
        Self::new(x0, f, comparator, method, g, eps, max_iters).result()
    }
    pub fn Mnimimum(
        x0: VectorN<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        method: Method,
        g: Vec<Bound<Scalar, Dimension>>,
        eps: Scalar,
        max_iters: usize,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        use crate::searchers::extremum_searcher::Search;
        Self::new(x0, f, std::cmp::Ordering::Less, method, g, eps, max_iters).result()
    }
    pub fn Maximum(
        x0: VectorN<Scalar, Dimension>,
        f: Arc<dyn Fn(VectorN<Scalar, Dimension>) -> Scalar>,
        method: Method,
        g: Vec<Bound<Scalar, Dimension>>,
        eps: Scalar,
        max_iters: usize,
    ) -> FinalResult<VectorN<Scalar, Dimension>> {
        use crate::searchers::extremum_searcher::Search;
        Self::new(
            x0,
            f,
            std::cmp::Ordering::Greater,
            method,
            g,
            eps,
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
        if self.g.iter().any(|g| {
            let _g = (g.function)(self.x.clone());
            match g.bound_type {
                BoundType::Equal => _g.abs() >= self.eps,
                BoundType::Unequal => _g >= self.eps,
            }
        }) && self
            .g
            .iter()
            .filter(|g| {
                let _g = (g.function)(self.x.clone());
                match g.bound_type {
                    BoundType::Equal => _g.abs() >= self.eps,
                    BoundType::Unequal => _g >= self.eps,
                }
            })
            .all(|r| !r.coefficient.is_finite())
        {
            return None;
        }
        let g = self.g.clone();
        let f = self.f.clone();
        let result = descent_methods::Search::result(
            self.x.clone(),
            Arc::new(move |x: VectorN<Scalar, Dimension>| -> Scalar {
                g.iter().fold(f(x.clone()), |result, i| {
                    result + i.coefficient * (i.penalty)((i.function)(x.clone()))
                })
            }),
            self.comparator,
            self.method.clone(),
            self.eps,
            self.max_iters,
        );
        self.func_calls += result.func_calls();
        let x = self.x.clone();
        self.x = result.x();
        self.dx = self.x.clone() - x;
        let x = self.x.clone();
        let eps = self.eps;
        let is_extra = (self
            .g
            .iter_mut()
            .filter(|g| {
                let _g = (g.function)(x.clone());
                g.coefficient.is_finite()
                    && match g.bound_type {
                        BoundType::Equal => _g.abs() >= eps,
                        BoundType::Unequal => _g >= eps,
                    }
            })
            .fold(self.got_result, |_, r| {
                r.coefficient = (r.coefficient_function)(r.coefficient);
                false
            })
        )//&& self.iters > 0)
            || self.iters >= self.max_iters;
        self.got_result = self.g.iter().all(|g| {
            let _g = (g.function)(self.x.clone());
            match g.bound_type {
                BoundType::Equal => _g.abs() < self.eps,
                BoundType::Unequal => _g < self.eps,
            }
        });
        self.iters += 1;
        return Some(Self::Item::new(
            self.x.clone(),
            self.dx.clone(),
            result.func_calls(),
            is_extra,
        ));
    }
}

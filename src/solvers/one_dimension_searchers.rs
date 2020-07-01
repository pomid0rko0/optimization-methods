use super::extremum_searcher;
use super::extremum_searcher::{FinalResult, IterationResult};
use nalgebra::RealField;
use std::sync::Arc;

fn get_interval<Scalar>(
    f: Arc<dyn Fn(Scalar) -> Scalar>,
    comparator: std::cmp::Ordering,
    x0: Scalar,
    delta: Scalar,
) -> (Scalar, Scalar, usize)
where
    Scalar: RealField,
{
    let mut h;
    let mut x = x0;
    let mut func_calls = 2;
    let mut f1 = f(x);
    let mut f2 = f(x + delta);

    if f2.partial_cmp(&f1).unwrap() == comparator {
        h = delta;
    } else {
        func_calls += 1;
        f2 = f(x - delta);
        if f2.partial_cmp(&f1).unwrap() == comparator {
            h = -delta;
        } else {
            return (x - delta, x + delta, 3);
        }
    }
    x = x + h;
    while f2.partial_cmp(&f1).unwrap() == comparator {
        h = h * Scalar::from_i8(2).unwrap();
        x = x + h;
        f1 = f2;
        f2 = f(x);
        func_calls += 1;
    }
    let left = (x - Scalar::from_f64(3. / 2.).unwrap() * h).min(x);
    let right = (x - Scalar::from_f64(3. / 2.).unwrap() * h).max(x);
    (left, right, func_calls)
}

trait OneDimensionalSearcher<Scalar>: extremum_searcher::Search<Scalar>
where
    Scalar: Clone,
{
}

#[derive(Clone)]
struct Fibonacci<Scalar> {
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
    fn new(
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

pub enum Method {
    Fibonacci,
}

pub struct Search<Scalar>
where
    Scalar: RealField,
{
    x: Scalar,
    dx: Scalar,
    func_calls: usize,
    iters: usize,
    method: Box<dyn extremum_searcher::Search<Scalar>>,
}

impl<Scalar> Search<Scalar>
where
    Scalar: RealField,
{
    pub fn new(
        x0: Scalar,
        f: Arc<dyn Fn(Scalar) -> Scalar>,
        comparator: std::cmp::Ordering,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> Self {
        let (left, right, func_calls) = get_interval(f.clone(), comparator, x0, eps);
        let m = match method {
            Method::Fibonacci => Fibonacci::new(left, right, f, comparator, eps, max_iters),
        };
        Self {
            x: (right + left) / Scalar::from_i8(2).unwrap(),
            dx: Scalar::max_value(),
            func_calls: func_calls + m.func_calls,
            iters: 0,
            method: Box::new(m),
        }
    }
    pub fn result(
        x0: Scalar,
        f: Arc<dyn Fn(Scalar) -> Scalar>,
        comparator: std::cmp::Ordering,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> FinalResult<Scalar> {
        use crate::solvers::extremum_searcher::Search;
        Self::new(x0, f, comparator, method, eps, max_iters).result()
    }
    pub fn Mnimimum(
        x0: Scalar,
        f: Arc<dyn Fn(Scalar) -> Scalar>,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> FinalResult<Scalar> {
        use crate::solvers::extremum_searcher::Search;
        Self::new(x0, f, std::cmp::Ordering::Less, method, eps, max_iters).result()
    }
    pub fn Maximum(
        x0: Scalar,
        f: Arc<dyn Fn(Scalar) -> Scalar>,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> FinalResult<Scalar> {
        use crate::solvers::extremum_searcher::Search;
        Self::new(x0, f, std::cmp::Ordering::Greater, method, eps, max_iters).result()
    }
}

impl<Scalar> extremum_searcher::Search<Scalar> for Search<Scalar>
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
        self.x
    }
    fn dx(&self) -> Scalar {
        self.dx
    }
}

impl<Scalar> Iterator for Search<Scalar>
where
    Scalar: RealField,
{
    type Item = IterationResult<Scalar>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.method.next() {
            Some(r) => {
                self.iters += 1;
                self.x = r.x();
                self.dx = r.dx();
                self.func_calls += r.func_calls();
                Some(IterationResult::new(
                    self.x,
                    self.dx,
                    r.func_calls(),
                    r.is_extra(),
                ))
            }
            None => None,
        }
    }
}

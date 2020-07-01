use crate::searchers::extremum_searcher;
use crate::searchers::extremum_searcher::{FinalResult, IterationResult};

pub trait OneDimensionalSearcher<Scalar>: extremum_searcher::Search<Scalar>
where
    Scalar: Clone,
{
}

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
            Method::Fibonacci => {
                super::fibonacci::Fibonacci::new(left, right, f, comparator, eps, max_iters)
            }
        };
        use crate::searchers::extremum_searcher::Search;
        Self {
            x: (right + left) / Scalar::from_i8(2).unwrap(),
            dx: Scalar::max_value(),
            func_calls: func_calls + m.func_calls(),
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
        use crate::searchers::extremum_searcher::Search;
        Self::new(x0, f, comparator, method, eps, max_iters).result()
    }
    pub fn Mnimimum(
        x0: Scalar,
        f: Arc<dyn Fn(Scalar) -> Scalar>,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> FinalResult<Scalar> {
        use crate::searchers::extremum_searcher::Search;
        Self::new(x0, f, std::cmp::Ordering::Less, method, eps, max_iters).result()
    }
    pub fn Maximum(
        x0: Scalar,
        f: Arc<dyn Fn(Scalar) -> Scalar>,
        method: Method,
        eps: Scalar,
        max_iters: usize,
    ) -> FinalResult<Scalar> {
        use crate::searchers::extremum_searcher::Search;
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

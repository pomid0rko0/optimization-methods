pub struct IterationResult<X> {
    x: X,
    dx: X,
    func_calls: usize,
    is_extra: bool,
}

impl<X> IterationResult<X>
where
    X: Clone,
{
    pub fn new(x: X, dx: X, func_calls: usize, is_extra: bool) -> Self {
        Self {
            x,
            dx,
            func_calls,
            is_extra,
        }
    }

    pub fn x(&self) -> X {
        self.x.clone()
    }
    pub fn dx(&self) -> X {
        self.dx.clone()
    }
    pub fn func_calls(&self) -> usize {
        self.func_calls
    }
    pub fn is_extra(&self) -> bool {
        self.is_extra
    }
}

pub struct FinalResult<X> {
    x: X,
    iters: usize,
    func_calls: usize,
}

impl<X> FinalResult<X>
where
    X: Clone,
{
    pub fn new(x: X, iters: usize, func_calls: usize) -> Self {
        Self {
            x,
            iters,
            func_calls,
        }
    }

    pub fn x(&self) -> X {
        self.x.clone()
    }
    pub fn func_calls(&self) -> usize {
        self.func_calls
    }
    pub fn iters(&self) -> usize {
        self.iters
    }
}

pub trait Search<X: Clone>: Iterator<Item = IterationResult<X>> {
    fn x(&self) -> X;
    fn dx(&self) -> X;
    fn func_calls(&self) -> usize;
    fn iters(&self) -> usize;
    fn result(&mut self) -> FinalResult<X> {
        let x = self.x().clone();
        let iters = self.iters();
        let func_calls = self.func_calls();
        self.take_while(|i| !i.is_extra()).fold(
            FinalResult::new(x, iters, func_calls),
            |result, i| {
                FinalResult::new(i.x(), result.iters + 1, result.func_calls + i.func_calls())
            },
        )
    }
}

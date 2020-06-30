/*
use nalgebra::allocator::Allocator;
use nalgebra::DefaultAllocator;
use nalgebra::DimName;
use nalgebra::MatrixN;
use nalgebra::VectorN;

use super::one_dimension_searchers::minimize;

pub fn broyden<D>(
    f: &dyn Fn(&VectorN<f64, D>) -> f64,
    df: &dyn Fn(&VectorN<f64, D>) -> VectorN<f64, D>,
    mut x: VectorN<f64, D>,
    eps: f64,
    rev: bool,
) -> (VectorN<f64, D>, i32, i32, String)
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D> + Allocator<f64, nalgebra::U1, D>,
{
    let mut iter = 0;
    let mut func_calls = 1;
    let mut result = String::new();

    let precision = -eps.log10().round() as usize;
    let width = precision + 6;
    result.push_str(&format!("\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\n",
        "i", "xi", "yi", "f(x, y)", "si1", "si2", "lambdai", "|yi-y(i-1)|", "|yi-y(i-1)|", "|fi-f(i-1)|", "angle((xi, yi), si)", "gi1", "gi2", "etai11", "etai12", "etai21", "etai22"));

    let mut g = df(&x);
    let mut eta = MatrixN::<f64, D>::from_diagonal_element(1.);

    result.push_str(&format!("\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\n",
    iter,
    x[0],
    x[1],
    if rev { 1./ f(&x) } else {f(&x)},
    (&eta * &g)[0],
    (&eta * &g)[1],
    0,
    0,
    0,
    0,
    x.angle(&(&eta * &g)),
    g[0],
    g[1],
    eta[(0, 0)],
    eta[(0, 1)],
    eta[(1, 0)],
    eta[(1, 1)]));

    loop {
        #[allow(non_snake_case)]
        let S = &eta * &g;
        let (lambda, search_func_calls) =
            minimize(&|lambda: f64| -> f64 { f(&(&x + lambda * &S)) }, 0., eps);
        let dx = &(lambda * &S);
        x += dx;
        iter += 1;
        func_calls += search_func_calls;
        let dg = &(df(&x) - &g);
        g += dg;
        func_calls += 1;

        if iter % D::dim() as i32 == 0 {
            eta = MatrixN::<f64, D>::from_diagonal_element(1.);
        } else {
            let dx_eta_dg = &(dx - &eta * dg);
            eta += dx_eta_dg * dx_eta_dg.transpose() / dx_eta_dg.dot(dg);
        }

        result.push_str(&format!("\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\"{}\";\n",
        iter,
        x[0],
        x[1],
        if rev { 1./ f(&x) } else {f(&x)},
        S[0],
        S[1],
        lambda,
        dx[0].abs(),
        dx[1].abs(),
        (f(&x) - f(&(&x - dx))).abs(),
        x.angle(&S),
        g[0],
        g[1],
        eta[(0, 0)],
        eta[(0, 1)],
        eta[(1, 0)],
        eta[(1, 1)]));

        if g.norm() < eps {
            return (x, func_calls, iter, result);
        }
    }
}
*/

#![allow(non_snake_case)]

use std::fs;
use std::sync::Arc;

use nalgebra::Vector2;

mod solvers;
use solvers::descent_methods::Method;
use solvers::minimize::FinalResult;
use solvers::penalty_methods::*;

//#region definitions

const STATS_HEADER: &'static str = "\"function calls\";\"iterations\";\"x\";\"y\";\"f(x, y)\";\n";

fn f(x: Vector2<f64>) -> f64 {
    5. * (x[0] - x[1]).powi(2) + (x[0] - 2.).powi(2)
}

fn g(x: Vector2<f64>) -> f64 {
    x[0] + x[1] - 1.
}

fn h(x: Vector2<f64>) -> f64 {
    x[0] + x[1]
}

fn G1(g: f64) -> f64 {
    0.5 * (g + g.abs())
}
fn G2(g: f64) -> f64 {
    (0.5 * (g + g.abs())).powi(2)
}
fn G3(g: f64) -> f64 {
    (0.5 * (g + g.abs())).powi(100)
}

fn H1(h: f64) -> f64 {
    h.abs()
}
fn H2(h: f64) -> f64 {
    h.powi(2)
}
fn H3(h: f64) -> f64 {
    h.powi(10)
}

fn R1(r: f64) -> f64 {
    r * 2.
}
fn R2(r: f64) -> f64 {
    r + r / (1. + r)
}
fn R3(r: f64) -> f64 {
    r.powi(2) * 2.
}

fn _G1(g: f64) -> f64 {
    if g > 0. {
        f64::MAX
    } else {
        -1. / g
    }
}
fn _G2(g: f64) -> f64 {
    if g > 0. {
        f64::MAX
    } else {
        (-1. / g).powi(3)
    }
}
fn _G3(g: f64) -> f64 {
    if g > 0. {
        f64::MAX
    } else {
        -(-g).ln()
    }
}

fn _R1(r: f64) -> f64 {
    r / 2.
}
fn _R2(r: f64) -> f64 {
    r - r / (1. + r)
}
fn _R3(r: f64) -> f64 {
    r.sin().abs()
}

fn format_result(result: FinalResult<Vector2<f64>>, eps: f64) -> String {
    let x = result.x();
    let func_calls = result.func_calls();
    let iters = result.iters();
    format!(
        "\"{}\";\"{}\";\"{:.prec$}\";\"{:.prec$}\";\"{:.prec$}\";\n",
        func_calls,
        iters,
        x[0],
        x[1],
        f(x),
        prec = -eps.log10().round() as usize
    )
    .replace(".", ",")
}

//#endregion

fn main() -> std::io::Result<()> {
    //#region init

    let max_iters: usize = 10000;
    let x0 = [
        Vector2::<f64>::from([0., 0.]),
        Vector2::<f64>::from([1., 1.]),
        Vector2::<f64>::from([-100., -100.]),
    ];
    let G: [Arc<dyn Fn(f64) -> f64>; 3] = [Arc::new(G1), Arc::new(G2), Arc::new(G3)];
    let H: [Arc<dyn Fn(f64) -> f64>; 3] = [Arc::new(H1), Arc::new(H2), Arc::new(H3)];
    let r0 = [1., 1f64.powi(-32), 1f64.powi(32)];
    let R: [Arc<dyn Fn(f64) -> f64>; 3] = [Arc::new(R1), Arc::new(R2), Arc::new(R3)];
    let eps: [f64; 3] = [1e-3, 1e-5, 1e-7];
    let mut stats: String;

    //#endregion

    //#region penalty a

    stats = String::from(STATS_HEADER);
    for Gi in G.iter() {
        stats += &format_result(
            Minimize::result(
                x0[0],
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(g),
                    BoundType::Unequal,
                    Gi.clone(),
                    r0[0],
                    R[0].clone(),
                )],
                eps[0],
                max_iters,
            ),
            eps[0],
        );
    }
    stats += ";\n";
    for &r0i in r0.iter() {
        stats += &format_result(
            Minimize::result(
                x0[0],
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(g),
                    BoundType::Unequal,
                    G[0].clone(),
                    r0i,
                    R[0].clone(),
                )],
                eps[0],
                max_iters,
            ),
            eps[0],
        );
    }
    stats += ";\n";
    for Ri in R.iter() {
        stats += &format_result(
            Minimize::result(
                x0[0],
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(g),
                    BoundType::Unequal,
                    G[0].clone(),
                    r0[0],
                    Ri.clone(),
                )],
                eps[0],
                max_iters,
            ),
            eps[0],
        );
    }
    stats += ";\n";
    for &x0i in x0.iter() {
        stats += &format_result(
            Minimize::result(
                x0i,
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(g),
                    BoundType::Unequal,
                    G[0].clone(),
                    r0[0],
                    R[0].clone(),
                )],
                eps[0],
                max_iters,
            ),
            eps[0],
        );
    }
    stats += ";\n";
    for &epsi in eps.iter() {
        stats += &format_result(
            Minimize::result(
                x0[0],
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(g),
                    BoundType::Unequal,
                    G[0].clone(),
                    r0[0],
                    R[0].clone(),
                )],
                epsi,
                max_iters,
            ),
            epsi,
        );
    }
    fs::write("lab3/penalty_methods_a.csv", &stats)?;

    //#endregion

    //#region penalty b

    stats = String::from(STATS_HEADER);
    for Hi in H.iter() {
        stats += &format_result(
            Minimize::result(
                x0[0],
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(h),
                    BoundType::Equal,
                    Hi.clone(),
                    r0[0],
                    R[0].clone(),
                )],
                eps[0],
                max_iters,
            ),
            eps[0],
        );
    }
    stats += ";\n";
    for &r0i in r0.iter() {
        stats += &format_result(
            Minimize::result(
                x0[0],
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(h),
                    BoundType::Equal,
                    H[0].clone(),
                    r0i,
                    R[0].clone(),
                )],
                eps[0],
                max_iters,
            ),
            eps[0],
        );
    }
    stats += ";\n";
    for Ri in R.iter() {
        stats += &format_result(
            Minimize::result(
                x0[0],
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(h),
                    BoundType::Equal,
                    H[0].clone(),
                    r0[0],
                    Ri.clone(),
                )],
                eps[0],
                max_iters,
            ),
            eps[0],
        );
    }
    stats += ";\n";
    for &x0i in x0.iter() {
        stats += &format_result(
            Minimize::result(
                x0i,
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(h),
                    BoundType::Equal,
                    H[0].clone(),
                    r0[0],
                    R[0].clone(),
                )],
                eps[0],
                max_iters,
            ),
            eps[0],
        );
    }
    stats += ";\n";
    for &epsi in eps.iter() {
        stats += &format_result(
            Minimize::result(
                x0[0],
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(h),
                    BoundType::Equal,
                    H[0].clone(),
                    r0[0],
                    R[0].clone(),
                )],
                epsi,
                max_iters,
            ),
            epsi,
        );
    }
    fs::write("lab3/penalty_methods_b.csv", &stats)?;

    //#endregion

    //#region boundary a

    let _G: [Arc<dyn Fn(f64) -> f64>; 3] = [Arc::new(_G1), Arc::new(_G2), Arc::new(_G3)];
    let _R: [Arc<dyn Fn(f64) -> f64>; 3] = [Arc::new(_R1), Arc::new(_R2), Arc::new(_R3)];

    stats = String::from(STATS_HEADER);
    for _Gi in _G.iter() {
        stats += &format_result(
            Minimize::result(
                x0[0],
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(g),
                    BoundType::Unequal,
                    _Gi.clone(),
                    r0[0],
                    _R[0].clone(),
                )],
                eps[0],
                max_iters,
            ),
            eps[0],
        );
    }
    stats += ";\n";
    for &r0i in r0.iter() {
        stats += &format_result(
            Minimize::result(
                x0[0],
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(g),
                    BoundType::Unequal,
                    _G[0].clone(),
                    r0i,
                    _R[0].clone(),
                )],
                eps[0],
                max_iters,
            ),
            eps[0],
        );
    }
    stats += ";\n";
    for _Ri in _R.iter() {
        stats += &format_result(
            Minimize::result(
                x0[0],
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(g),
                    BoundType::Unequal,
                    _G[0].clone(),
                    r0[0],
                    _Ri.clone(),
                )],
                eps[0],
                max_iters,
            ),
            eps[0],
        );
    }
    stats += ";\n";
    for &x0i in x0.iter() {
        stats += &format_result(
            Minimize::result(
                x0i,
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(g),
                    BoundType::Unequal,
                    _G[0].clone(),
                    r0[0],
                    _R[0].clone(),
                )],
                eps[0],
                max_iters,
            ),
            eps[0],
        );
    }
    stats += ";\n";
    for &epsi in eps.iter() {
        stats += &format_result(
            Minimize::result(
                x0[0],
                Arc::new(f),
                Method::Gauss,
                vec![Bound::new(
                    Arc::new(g),
                    BoundType::Unequal,
                    _G[0].clone(),
                    r0[0],
                    _R[0].clone(),
                )],
                epsi,
                max_iters,
            ),
            epsi,
        );
    }
    fs::write("lab3/boundary_methods_a.csv", &stats)?;

    //#endregion
    Ok(())
}

#![allow(non_snake_case)]

use nalgebra::Vector2;

mod searchers;
use searchers::descent_methods::*;
use searchers::variable_metric_methods::*;

use std::fs;

/*
fn ddf(X: Vector) -> Matrix {
    let mut result = Matrix::new(2);
    let x = X[0];
    let y = X[1];
    let mut x1 = -1. / 9. * (x - 2.).powi(2);
    let mut y1 = -0.25 * (y - 3.).powi(2);
    let mut x2 = -1. / 9. * (x - 2.).powi(2);
    let mut y2 = -0.25 * (y - 1.).powi(2);
    let mut x3 = -0.25 * (x - 1.).powi(2);
    let mut y3 = -(y - 1.).powi(2);
    let x4 = -0.25 * (x - 1.).powi(2);
    let y4 = -(y - 1.).powi(2);
    let mut e1 = (x1 + y1).exp();
    let mut e2 = (x2 + y2).exp();
    let mut e3 = (x3 + y3).exp();
    let e4 = (x4 + y4).exp();
    result[0][0] =
        4. / 27. * (x - 2.).powi(2) * e1 - 2. / 3. * e2 - e3 + 0.5 * (x - 1.).powi(2) * e4;
    x1 = -1. / 9. * (x - 2.).powi(2);
    y1 = -0.25 * (y - 3.).powi(2);
    x2 = -0.25 * (x - 1.).powi(2);
    y2 = -(y - 1.).powi(2);
    e1 = (x1 + y1).exp();
    e2 = (x2 + y2).exp();
    result[0][1] = 1. / 3. * (x - 2.) * (y - 3.) * e1 + 2. * (x - 1.) * (y - 1.) * e2;
    result[1][0] = result[0][1];
    x2 = -0.25 * (x - 1.).powi(2);
    y2 = (y - 1.).powi(2);
    x3 = -1. / 9. * (x - 2.).powi(2);
    y3 = 0.25 * (y - 3.).powi(2);
    e1 = (1. / 36. * (-13. * x * x + 34. * x - 45. * y * y + 126. * y - 142.)).exp();
    e2 = (x2 + y2).exp();
    e3 = (x3 + y3).exp();
    result[1][1] =
        0.25 * e1 * (3. * (y * y - 6. * y + 7.) * e2 + 16. * (2. * y * y - 4. * y + 1.) * e3);
    result
}
*/

fn f(X: &Vector2<f64>) -> f64 {
    100. * (X[1] - X[0]).powi(2) + (1. - X[0]).powi(2)
}

fn df(X: &Vector2<f64>) -> Vector2<f64> {
    Vector2::new(-200. * X[1] + 202. * X[0] - 2., 200. * (X[1] - X[0]))
}

fn g(X: &Vector2<f64>) -> f64 {
    100. * (X[1] - X[0].powi(2)).powi(2) + (1. - X[0]).powi(2)
}

fn dg(X: &Vector2<f64>) -> Vector2<f64> {
    Vector2::new(
        2. * (-200. * X[1] * X[0] + 200. * X[0].powi(3) - 1. + X[0]),
        200. * (X[1] - X[0].powi(2)),
    )
}

fn h(X: &Vector2<f64>) -> f64 {
    let x = X[0];
    let y = X[1];
    let x1 = -((x - 1.) / 2.).powi(2);
    let y1 = -((y - 1.) / 1.).powi(2);
    let x2 = -((x - 2.) / 3.).powi(2);
    let y2 = -((y - 3.) / 2.).powi(2);
    let e1 = 2. * (x1 + y1).exp();
    let e2 = 3. * (x2 + y2).exp();
    e1 + e2
}

fn dh(X: &Vector2<f64>) -> Vector2<f64> {
    let x = X[0];
    let y = X[1];

    let x_x1 = -1. / 9. * (x - 2.).powi(2);
    let x_y1 = -0.25 * (y - 3.).powi(2);
    let x_x2 = -0.25 * (x - 1.).powi(2);
    let x_y2 = -(y - 1.).powi(2);
    let x_e1 = (x_x1 + x_y1).exp();
    let x_e2 = (x_x2 + x_y2).exp();

    let y_x1 = -1. / 9. * (x - 2.).powi(2);
    let y_y1 = -0.25 * (y - 3.).powi(2);
    let y_x2 = -0.25 * (x - 1.).powi(2);
    let y_y2 = -(y - 1.).powi(2);
    let y_e1 = (y_x1 + y_y1).exp();
    let y_e2 = (y_x2 + y_y2).exp();

    Vector2::new(
        -2. / 3. * (x - 2.) * x_e1 - (x - 1.) * x_e2,
        -3. / 2. * (y - 3.) * y_e1 - 4. * (y - 1.) * y_e2,
    )
}
/*
fn _f(X: &Vector2<f64>) -> f64 {
    1. / f(X)
}

fn _df(X: &Vector2<f64>) -> Vector2<f64> {
    -df(X) / f(X).powi(2)
}

fn _g(X: &Vector2<f64>) -> f64 {
    1. / g(X)
}

fn _dg(X: &Vector2<f64>) -> Vector2<f64> {
    -dg(X) / g(X).powi(2)
}
*/
fn _h(X: &Vector2<f64>) -> f64 {
    1. / h(X)
}

fn _dh(X: &Vector2<f64>) -> Vector2<f64> {
    -_h(X).powi(2) * dh(X)
}

fn main() -> std::io::Result<()> {
    let x0 = Vector2::new(-0.1, 0.1);
    //let x0 = Vector2::new(7., 7.);
    //let eps = 1e-7;
    //let width = precision + 6;

    let mut x;
    let mut iter;
    let mut func_calls;
    let mut details;

    let mut result;

    let mut stats =
        vec![String::from("\"function calls\";\"iterations\";\"x\";\"y\";\"f(x, y)\";\n"); 6];

    for (&eps, i) in [1e-3f64, 1e-4f64, 1e-5f64, 1e-6f64, 1e-7f64]
        .iter()
        .zip(3..8)
    {
        let precision = -eps.log10().round() as usize;

        //(x, iter, func_calls, details) = conjugate_gradients(&f, &df, x0, eps);
        result = conjugate_gradients(&f, &df, x0, eps);
        x = result.0;
        iter = result.1;
        func_calls = result.2;
        details = result.3;
        fs::write(
            format!("lab2/conjugate_gradients_details_f_{}.csv", i),
            details,
        )?;
        stats[0].push_str(&format!(
            "\"{:.prec$}\";\"{}\";\"{}\";\"{:.prec$}\";\"{:.prec$}\";\"{:.prec$}\";\n",
            eps,
            iter,
            func_calls,
            x[0],
            x[1],
            f(&x),
            prec = precision
        ));
        let precision = -eps.log10().round() as usize;

        result = conjugate_gradients(&g, &dg, x0, eps);
        x = result.0;
        iter = result.1;
        func_calls = result.2;
        details = result.3;
        fs::write(
            format!("lab2/conjugate_gradients_details_g_{}.csv", i),
            details,
        )?;
        stats[1].push_str(&format!(
            "\"{:.prec$}\";\"{}\";\"{}\";\"{:.prec$}\";\"{:.prec$}\";\"{:.prec$}\";\n",
            eps,
            iter,
            func_calls,
            x[0],
            x[1],
            g(&x),
            prec = precision
        ));

        result = conjugate_gradients(&_h, &_dh, x0, eps);
        x = result.0;
        iter = result.1;
        func_calls = result.2;
        details = result.3;
        fs::write(
            format!("lab2/conjugate_gradients_details_h_{}.csv", i),
            details,
        )?;
        stats[2].push_str(&format!(
            "\"{:.prec$}\";\"{}\";\"{}\";\"{:.prec$}\";\"{:.prec$}\";\"{:.prec$}\";\n",
            eps,
            iter,
            func_calls,
            x[0],
            x[1],
            h(&x),
            prec = precision
        ));

        result = broyden(&f, &df, x0, eps);
        x = result.0;
        iter = result.1;
        func_calls = result.2;
        details = result.3;
        fs::write(format!("lab2/broyden_details_f_{}.csv", i), details)?;
        stats[3].push_str(&format!(
            "\"{:.prec$}\";\"{}\";\"{}\";\"{:.prec$}\";\"{:.prec$}\";\"{:.prec$}\";\n",
            eps,
            iter,
            func_calls,
            x[0],
            x[1],
            f(&x),
            prec = precision
        ));
        let precision = -eps.log10().round() as usize;

        result = broyden(&g, &dg, x0, eps);
        x = result.0;
        iter = result.1;
        func_calls = result.2;
        details = result.3;
        fs::write(format!("lab2/broyden_details_g_{}.csv", i), details)?;
        stats[4].push_str(&format!(
            "\"{:.prec$}\";\"{}\";\"{}\";\"{:.prec$}\";\"{:.prec$}\";\"{:.prec$}\";\n",
            eps,
            iter,
            func_calls,
            x[0],
            x[1],
            g(&x),
            prec = precision
        ));

        result = broyden(&_h, &_dh, x0, eps);
        x = result.0;
        iter = result.1;
        func_calls = result.2;
        details = result.3;
        fs::write(format!("lab2/broyden_details_h_{}.csv", i), details)?;
        stats[5].push_str(&format!(
            "\"{:.prec$}\";\"{}\";\"{}\";\"{:.prec$}\";\"{:.prec$}\";\"{:.prec$}\";\n",
            eps,
            iter,
            func_calls,
            x[0],
            x[1],
            h(&x),
            prec = precision
        ));
    }
    fs::write("lab2/conjugate_gradients_stats_f.csv", &stats[0])?;
    fs::write("lab2/conjugate_gradients_stats_g.csv", &stats[1])?;
    fs::write("lab2/conjugate_gradients_stats_h.csv", &stats[2])?;
    fs::write("lab2/broyden_stats_f.csv", &stats[3])?;
    fs::write("lab2/broyden_stats_g.csv", &stats[4])?;
    fs::write("lab2/broyden_stats_h.csv", &stats[5])?;
    Ok(())
    /*
        let mut x;
        let mut iter;
        let mut func_calls;
        let mut result;
        //let x1 = Vector2::new(7., -5.);
        println!("\n\n###########################\n\n");
        (x, iter, func_calls, result) = conjugate_gradients(&f, &df, x1, eps);

        println!("----------------------");
        println!("{:?};", conjugate_gradients(&g, &dg, x1, eps));
        println!("----------------------");
        println!("{:?};", conjugate_gradients(&_h, &_dh, x1, eps));
        println!("\n\n###########################\n\n");
        println!("{:?};", broyden(&f, &df, x1, eps));
        println!("----------------------");
        println!("{:?};", broyden(&g, &dg, x1, eps));
        println!("----------------------");
        println!("{:?};", broyden(&_h, &_dh, x1, eps));
        println!("\n\n###########################\n\n");
    */
}

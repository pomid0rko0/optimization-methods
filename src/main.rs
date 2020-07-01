#![allow(non_snake_case)]

mod searchers {
    mod descent_searchers {
        mod gauss;
        pub mod search;
    }
    mod extremum_searcher;
    mod one_dimension_searchers {
        mod fibonacci;
        pub mod search;
    }
    mod random_searchers {
        mod global {
            mod first;
            mod second;
            mod third;
        };
        mod simple;
        mod search;
    }
}

use std::fs;
use std::sync::Arc;

use nalgebra::Vector2;

//#region definitions
//#endregion

const STATS_HEADER: &'static str = "\"function calls\";\"iterations\";\"x\";\"y\";\"f(x, y)\";\n";

fn main() -> std::io::Result<()> {
    Ok(())
}

pub mod card;
pub mod eval;
pub mod ev;
pub mod state;
pub mod abstraction;

#[cfg(feature = "pyo3-bindings")]
mod python;

#[cfg(feature = "pyo3-bindings")]
pub use python::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_compiles() {
        assert_eq!(card::NUM_CARDS, 52);
    }
}

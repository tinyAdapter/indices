use std::collections::HashMap;

use log::error;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

pub fn run_python_function(
    py_module: &Lazy<Py<PyModule>>,
    parameters: &String,
    function_name: &str,
) -> serde_json::Value {
    let parameters_str = parameters.to_string();
    let results = Python::with_gil(|py| -> String {
        let run_script: Py<PyAny> = py_module.getattr(py, function_name).unwrap().into();
        let result = run_script.call1(py, PyTuple::new(py, &[parameters_str.into_py(py)]));
        let result = match result {
            Err(e) => {
                let traceback = e.traceback(py).unwrap().format().unwrap();
                error!("{traceback} {e}");
                format!("{traceback} {e}")
            }
            Ok(o) => o.extract(py).unwrap(),
        };
        result
    });

    serde_json::from_str(&results).unwrap()
}

/*
Python Module Path for Model Selection
*/
pub static PY_MODULE: Lazy<Py<PyModule>> = Lazy::new(|| {
    Python::with_gil(|py| -> Py<PyModule> {
        let src = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            // was "/../ml/model_selection/pg_interface.py" but the folder does not exist
            // temp solution to avoid compiling error
            "/../ml/model_slicing/pg_interface.py"
        ));
        PyModule::from_code(py, src, "", "").unwrap().into()
    })
});

/*
Python Module Path for SAMS
*/
pub static PY_MODULE_INFERENCE: Lazy<Py<PyModule>> = Lazy::new(|| {
    Python::with_gil(|py| -> Py<PyModule> {
        let src = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../ml/model_slicing/pg_interface.py"
        ));
        PyModule::from_code(py, src, "", "").unwrap().into()
    })
});

pub fn record_results_py(results: &HashMap<&str, f64>) {
    run_python_function(
        &PY_MODULE_INFERENCE,
        &serde_json::json!(results).to_string(),
        "records_results",
    );
}

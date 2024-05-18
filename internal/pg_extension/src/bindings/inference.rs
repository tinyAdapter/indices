use crate::bindings::ml_register::record_results_py;
use crate::bindings::ml_register::run_python_function;
use crate::bindings::ml_register::PY_MODULE_INFERENCE;
use crate::util;
use pgrx::prelude::*;
use serde_json::json;
use shared_memory::*;
use std::collections::HashMap;
use std::ffi::c_long;
use std::time::Instant;

pub struct Model<'a> {
    condition: &'a str,
    config_file: &'a str,
    col_cardinalities_file: &'a str,
    model_path: &'a str,
}

impl<'a> Model<'a> {
    pub fn new(
        condition: &'a str,
        config_file: &'a str,
        col_cardinalities_file: &'a str,
        model_path: &'a str,
    ) -> Model<'a> {
        Model {
            condition,
            config_file,
            col_cardinalities_file,
            model_path,
        }
    }

    pub fn init(&self) {
        // Step 1: load model and columns etc
        let mut task_map = HashMap::new();
        task_map.insert("where_cond", self.condition);
        task_map.insert("config_file", self.config_file);
        task_map.insert("col_cardinalities_file", self.col_cardinalities_file);
        task_map.insert("model_path", self.model_path);
        let task_json = json!(task_map).to_string();
        // here it cache a state
        run_python_function(
            &PY_MODULE_INFERENCE,
            &task_json,
            "model_inference_load_model",
        );
    }
}

fn num_columns(dataset: &str) -> Result<i32, ()> {
    match dataset {
        // assuming dataset is a String
        "frappe" => Ok(12),
        "adult" => Ok(15),
        "cvd" => Ok(13),
        "bank" => Ok(18),
        "census" => Ok(41 + 2),
        "credit" => Ok(23 + 2),
        "diabetes" => Ok(48 + 2),
        "hcdr" => Ok(69 + 2),
        _ => Err(()),
    }
}

pub struct InferenceRunner {
    dataset: String,
    condition: String,
    config_file: String,
    col_cardinalities_file: String,
    model_path: String,
    sql: String,
    batch_size: i32,
}

impl InferenceRunner {
    pub fn new(
        dataset: String,
        condition: String,
        config_file: String,
        col_cardinalities_file: String,
        model_path: String,
        sql: String,
        batch_size: i32,
    ) -> InferenceRunner {
        InferenceRunner {
            dataset,
            condition,
            config_file,
            col_cardinalities_file,
            model_path,
            sql,
            batch_size,
        }
    }

    fn copy_data(&self, tup_table: &serde_json::Value, mini_batch_json: &str) {
        // Set an identifier for the shared memory
        let shmem_name = "my_shared_memory";
        let my_shmem = ShmemConf::new()
            .size(tup_table.to_string().len())
            .os_id(shmem_name)
            .create()
            .unwrap();

        // Use unsafe to access and write to the raw memory
        let data_to_write = mini_batch_json.as_bytes();
        unsafe {
            // Get the raw pointer to the shared memory
            let shmem_ptr = my_shmem.as_ptr() as *mut u8;
            // Copy data into the shared memory
            std::ptr::copy_nonoverlapping(data_to_write.as_ptr(), shmem_ptr, data_to_write.len());
        }
    }

    pub fn run_shared_memory(&self) -> serde_json::Value {
        let mut response = HashMap::new();

        let overall_start_time = Instant::now();

        // Step 1: load model and columns etc
        let model_init_time = util::record_time(|_| {
            Model::new(
                &self.condition,
                &self.config_file,
                &self.col_cardinalities_file,
                &self.model_path,
            )
            .init()
        });
        response.insert("model_init_time", model_init_time);

        // Step 2: query data via SPI
        let (data_query_time, tup_table) = util::record_time_returns(|_| {
            let mut last_id = 0;

            let results: Result<Vec<Vec<String>>, String> = Spi::connect(|client| {
                let query = format!(
                    "SELECT * FROM {}_train {} LIMIT {}",
                    self.dataset, self.sql, self.batch_size
                );
                let mut cursor = client.open_cursor(&query, None);
                let table = match cursor.fetch(self.batch_size as c_long) {
                    Ok(table) => table,
                    Err(e) => return Err(e.to_string()), // Convert the error to a string and return
                };

                let mut mini_batch = Vec::new();

                for row in table.into_iter() {
                    let mut each_row = Vec::new();
                    // add primary key
                    let col0 = match row.get::<i32>(1) {
                        Ok(Some(val)) => {
                            // Update last_id with the retrieved value
                            if val > 100000 {
                                last_id = 0;
                            } else {
                                last_id = val
                            }
                            val.to_string()
                        }
                        Ok(None) => "".to_string(), // Handle the case when there's no valid value
                        Err(e) => e.to_string(),
                    };
                    each_row.push(col0);
                    // add label
                    let col1 = match row.get::<i32>(2) {
                        Ok(val) => val.map(|i| i.to_string()).unwrap_or_default(),
                        Err(e) => e.to_string(),
                    };
                    each_row.push(col1);
                    // add fields
                    let texts: Vec<String> = (3..row.columns() + 1)
                        .filter_map(|i| {
                            match row.get::<&str>(i) {
                                Ok(Some(s)) => Some(s.to_string()),
                                Ok(None) => None,
                                Err(e) => Some(e.to_string()), // Convert error to string
                            }
                        })
                        .collect();
                    each_row.extend(texts);
                    mini_batch.push(each_row)
                }
                // return
                Ok(mini_batch)
            });
            // serialize the mini-batch data
            match results {
                Ok(data) => {
                    serde_json::json!({
                        "status": "success",
                        "data": data
                    })
                }
                Err(e) => {
                    serde_json::json!({
                        "status": "error",
                        "message": format!("Error while connecting: {}", e)
                    })
                }
            }
        });
        response.insert("data_query_time", data_query_time);

        let mini_batch_json = tup_table.to_string();

        let data_copy_time = util::record_time(|_| {
            self.copy_data(&tup_table, &mini_batch_json);
        });
        response.insert("data_copy", data_copy_time);

        // Step 3: model evaluate in Python
        let python_compute_time = util::record_time(|_| {
            let mut eva_task_map = HashMap::new();
            eva_task_map.insert("config_file", self.config_file.clone());
            eva_task_map.insert("spi_seconds", data_query_time.to_string());

            let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

            run_python_function(
                &PY_MODULE_INFERENCE,
                &eva_task_json,
                "model_inference_compute_shared_memory",
            );
        });
        response.insert("python_compute_time", python_compute_time);

        let overall_elapsed_time = util::time_since(&overall_start_time);
        let diff_time = model_init_time + data_query_time + data_copy_time + python_compute_time
            - overall_elapsed_time;

        response.insert("overall_query_latency", overall_elapsed_time);
        response.insert("diff", diff_time);

        serde_json::json!(response)
    }

    pub fn run(&self) -> serde_json::Value {
        let mut response = HashMap::new();

        let overall_start_time = Instant::now();

        //     let mut last_id = 0;

        // Step 1: load model and columns etc
        let model_init_time = util::record_time(|_| {
            Model::new(
                &self.condition,
                &self.config_file,
                &self.col_cardinalities_file,
                &self.model_path,
            )
            .init()
        });

        response.insert("model_init_time", model_init_time);

        // Step 2: query data via SPI
        let mut all_rows = Vec::new();

        let data_query_time = util::record_time(|start_time| {
            let _ = Spi::connect(|client| {
                let query = format!(
                    "SELECT * FROM {}_int_train {} LIMIT {}",
                    self.dataset, self.sql, self.batch_size
                );
                let mut cursor = client.open_cursor(&query, None);
                let table = match cursor.fetch(self.batch_size as c_long) {
                    Ok(table) => table,
                    Err(e) => return Err(e.to_string()),
                };

                response.insert("data_query_time_spi", util::time_since(start_time));

                // todo: nl: this part can must be optimized, since i go through all of those staff.
                for row in table.into_iter() {
                    for i in 3..=row.columns() {
                        match row.get::<i32>(i) {
                            Ok(Some(val)) => all_rows.push(val), // Handle the case when a valid i32 is obtained
                            Ok(None) => {
                                // Handle the case when the value is missing or erroneous
                                // For example, you can add a default value, like -1
                                all_rows.push(-1);
                            }
                            Err(e) => {
                                // Handle the error, e.g., log it or handle it in some way
                                eprintln!("Error fetching value: {:?}", e);
                            }
                        }
                    }
                }
                // Return OK or some status
                Ok(())
            });
        });
        response.insert("data_query_time", data_query_time);

        let mini_batch_json = serde_json::to_string(&all_rows).unwrap();
        let mini_batch_json_str = mini_batch_json.as_str();

        // Step 3: model evaluate in Python
        let python_compute_time = util::record_time(|_| {
            let mut eva_task_map = HashMap::new();
            eva_task_map.insert("config_file", self.config_file.clone());
            eva_task_map.insert("mini_batch", mini_batch_json_str.to_string());
            eva_task_map.insert("spi_seconds", data_query_time.to_string());

            let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

            run_python_function(
                &PY_MODULE_INFERENCE,
                &eva_task_json,
                "model_inference_compute",
            );
        });
        response.insert("python_compute_time", python_compute_time);

        let overall_elapsed_time = util::time_since(&overall_start_time);
        let diff_time =
            model_init_time + data_query_time + python_compute_time - overall_elapsed_time;

        response.insert("overall_query_latency", overall_elapsed_time);
        response.insert("diff", diff_time);

        record_results_py(&response);

        serde_json::json!(response)
    }

    pub fn run_shared_memory_write_once(&self) -> serde_json::Value {
        let mut response = HashMap::new();

        let overall_start_time = Instant::now();

        let mut last_id = 0;

        // Step 1: load model and columns etc
        let model_init_time = util::record_time(|_| {
            Model::new(
                &self.condition,
                &self.config_file,
                &self.col_cardinalities_file,
                &self.model_path,
            )
            .init()
        });
        response.insert("model_init_time", model_init_time);

        // Step 2: query data via SPI
        let (mem_allocate_time, my_shmem) = util::record_time_returns(|_| {
            // Allocate shared memory in advance
            // Set an identifier for the shared memory
            let shmem_name = "my_shared_memory";

            // Pre-allocate a size for shared memory (this might need some logic to determine a reasonable size)
            let avg_row_size = 120;
            let shmem_size = (1.5 * (avg_row_size * self.batch_size as usize) as f64) as usize;
            ShmemConf::new()
                .size(shmem_size)
                .os_id(shmem_name)
                .create()
                .unwrap()
        });
        response.insert("mem_allocate_time", mem_allocate_time);

        let shmem_ptr = my_shmem.as_ptr() as *mut u8;
        let shmem_size = my_shmem.len();

        let data_query_time = util::record_time(|start_time| {
            let _ = Spi::connect(|client| {
                let query = format!(
                    "SELECT * FROM {}_train {} LIMIT {}",
                    self.dataset, self.sql, self.batch_size
                );
                let mut cursor = client.open_cursor(&query, None);
                let table = match cursor.fetch(self.batch_size as c_long) {
                    Ok(table) => table,
                    Err(e) => return Err(e.to_string()),
                };

                response.insert("data_query_time_spi", util::time_since(start_time));

                let mut offset = 0; // Keep track of how much we've written to shared memory

                // Write the opening square bracket
                unsafe {
                    shmem_ptr.offset(offset as isize).write(b"["[0]);
                }
                offset += 1;

                let mut is_first_row = true;
                for row in table.into_iter() {
                    // If not the first row, write a comma before the next row's data
                    if !is_first_row {
                        unsafe {
                            shmem_ptr.offset(offset as isize).write(b","[0]);
                        }
                        offset += 1;
                    } else {
                        is_first_row = false;
                    }

                    let mut each_row = Vec::new();
                    // add primary key
                    let col0 = match row.get::<i32>(1) {
                        Ok(Some(val)) => {
                            // Update last_id with the retrieved value
                            if val > 100000 {
                                last_id = 0;
                            } else {
                                last_id = val
                            }
                            val.to_string()
                        }
                        Ok(None) => "".to_string(), // Handle the case when there's no valid value
                        Err(e) => e.to_string(),
                    };
                    each_row.push(col0);
                    // add label
                    let col1 = match row.get::<i32>(2) {
                        Ok(val) => val.map(|i| i.to_string()).unwrap_or_default(),
                        Err(e) => e.to_string(),
                    };
                    each_row.push(col1);
                    // add fields
                    let texts: Vec<String> = (3..row.columns() + 1)
                        .filter_map(|i| {
                            match row.get::<&str>(i) {
                                Ok(Some(s)) => Some(s.to_string()),
                                Ok(None) => None,
                                Err(e) => Some(e.to_string()), // Convert error to string
                            }
                        })
                        .collect();
                    each_row.extend(texts);

                    // Serialize each row into shared memory
                    let serialized_row = serde_json::to_string(&each_row).unwrap();
                    let bytes = serialized_row.as_bytes();

                    // Check if there's enough space left in shared memory
                    if offset + bytes.len() > shmem_size {
                        // Handle error: not enough space in shared memory
                        return Err("Shared memory exceeded estimated size.".to_string());
                    }

                    // Copy the serialized row into shared memory
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            shmem_ptr.offset(offset as isize),
                            bytes.len(),
                        );
                    }
                    offset += bytes.len();
                }
                // Write the closing square bracket after all rows
                unsafe {
                    shmem_ptr.offset(offset as isize).write(b"]"[0]);
                }

                // Return OK or some status
                Ok(())
            });
        });
        response.insert("data_query_time", data_query_time);

        // Step 3: model evaluate in Python
        let python_compute_time = util::record_time(|_| {
            let mut eva_task_map = HashMap::new();
            eva_task_map.insert("config_file", self.config_file.clone());
            eva_task_map.insert("spi_seconds", data_query_time.to_string());

            let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

            run_python_function(
                &PY_MODULE_INFERENCE,
                &eva_task_json,
                "model_inference_compute_shared_memory_write_once",
            );
        });
        response.insert("python_compute_time", python_compute_time);

        let overall_elapsed_time = util::time_since(&overall_start_time);
        let diff_time =
            model_init_time + data_query_time + python_compute_time - overall_elapsed_time;

        response.insert("overall_query_latency", overall_elapsed_time);
        response.insert("diff", diff_time);

        record_results_py(&response);

        serde_json::json!(response)
    }

    pub fn run_shared_memory_write_once_int_exp(&self) -> serde_json::Value {
        let mut response = HashMap::new();
        // let mut response_log = HashMap::new();

        let num_columns: i32 = num_columns(&self.dataset).unwrap();

        let overall_start_time = Instant::now();

        /* load model and columns etc */
        response.insert(
            "model_init_time",
            util::record_time(|_| {
                Model::new(
                    &self.condition,
                    &self.config_file,
                    &self.col_cardinalities_file,
                    &self.model_path,
                )
                .init()
            }),
        );

        /* query data */
        let mut all_rows = Vec::new();

        let data_query_time = util::record_time(|start_time| {
            let _ = Spi::connect(|client| {
                let query = format!(
                    "SELECT * FROM {}_int_train {} LIMIT {}",
                    self.dataset, self.sql, self.batch_size
                );
                let mut cursor = client.open_cursor(&query, None);
                let table = match cursor.fetch(self.batch_size as c_long) {
                    Ok(table) => table,
                    Err(e) => return Err(e.to_string()),
                };

                response.insert("data_query_time_spi", util::time_since(start_time));

                let mut t1: f64 = 0.0;
                // todo: nl: this part can must be optimized, since i go through all of those staff.
                response.insert(
                    "data_query_time3",
                    util::record_time_move(|_| {
                        for row in table.into_iter() {
                            for i in 3..=num_columns as usize {
                                t1 += util::record_time(|_| {
                                    if let Ok(Some(val)) = row.get::<i32>(i) {
                                        all_rows.push(val);
                                    }
                                });
                            }
                        }
                    }),
                );
                response.insert("data_query_time2", t1);

                // Return OK or some status
                Ok(())
            });
        });
        response.insert("data_query_time", data_query_time);

        /* log the query datas */
        // let serialized_row = serde_json::to_string(&all_rows).unwrap();
        // response_log.insert("query_data", serialized_row);

        /* Putting all data to he shared memory */
        let mem_allocate_time = util::record_time(|_| {
            let shmem_name = "my_shared_memory";
            let my_shmem = ShmemConf::new()
                .size(4 * all_rows.len())
                .os_id(shmem_name)
                .create()
                .unwrap();
            let shmem_ptr = my_shmem.as_ptr() as *mut i32;

            // Copy data into shared memory
            unsafe {
                std::ptr::copy_nonoverlapping(
                    all_rows.as_ptr(),
                    shmem_ptr as *mut i32,
                    all_rows.len(),
                );
            }
        });
        response.insert("mem_allocate_time", mem_allocate_time);

        // Step 3: model evaluate in Python
        let python_compute_time = util::record_time(|_| {
            let mut eva_task_map = HashMap::new();
            eva_task_map.insert("config_file", self.config_file.clone());
            eva_task_map.insert("spi_seconds", data_query_time.to_string());
            eva_task_map.insert("rows", self.batch_size.to_string());

            let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

            run_python_function(
                &PY_MODULE_INFERENCE,
                &eva_task_json,
                "model_inference_compute_shared_memory_write_once_int",
            );
        });
        response.insert("python_compute_time", python_compute_time);

        let overall_elapsed_time = util::time_since(&overall_start_time);
        let diff_time = response["model_init_time"] + data_query_time + python_compute_time
            - overall_elapsed_time;

        response.insert("overall_query_latency", overall_elapsed_time);
        response.insert("diff", diff_time);

        record_results_py(&response);

        serde_json::json!(response)
    }

    pub fn run_shared_memory_write_once_int(&self) -> serde_json::Value {
        let mut response = HashMap::new();

        let num_columns: i32 = num_columns(&self.dataset).unwrap();

        let overall_start_time = Instant::now();

        // Step 1: load model and columns etc
        let model_init_time = util::record_time(|_| {
            Model::new(
                &self.condition,
                &self.config_file,
                &self.col_cardinalities_file,
                &self.model_path,
            )
            .init()
        });
        response.insert("model_init_time", model_init_time);

        // Step 1: query data
        let mut all_rows = Vec::new();

        let data_query_time = util::record_time(|start_time| {
            let _ = Spi::connect(|client| {
                let query = format!(
                    "SELECT * FROM {}_int_train {} LIMIT {}",
                    self.dataset, self.sql, self.batch_size
                );
                let mut cursor = client.open_cursor(&query, None);
                let table = match cursor.fetch(self.batch_size as c_long) {
                    Ok(table) => table,
                    Err(e) => return Err(e.to_string()),
                };

                response.insert("data_query_time_spi", util::time_since(start_time));

                // todo: nl: this part can must be optimized, since i go through all of those staff.
                response.insert(
                    "data_type_convert_time",
                    util::record_time_move(|_| {
                        for row in table.into_iter() {
                            for i in 3..=num_columns as usize {
                                if let Ok(Some(val)) = row.get::<i32>(i) {
                                    all_rows.push(val);
                                }
                            }
                        }
                    }),
                );

                // Return OK or some status
                Ok(())
            });
        });
        response.insert("data_query_time", data_query_time);

        // log the query datas
        // let serialized_row = serde_json::to_string(&all_rows).unwrap();
        // response_log.insert("query_data", serialized_row);

        // Step 3: Putting all data to he shared memory
        let mem_allocate_time = util::record_time(|_| {
            let shmem_name: &str = "my_shared_memory";
            let my_shmem = ShmemConf::new()
                .size(4 * all_rows.len())
                .os_id(shmem_name)
                .create()
                .unwrap();
            let shmem_ptr = my_shmem.as_ptr() as *mut i32;

            unsafe {
                // Copy data into shared memory
                std::ptr::copy_nonoverlapping(
                    all_rows.as_ptr(),
                    shmem_ptr as *mut i32,
                    all_rows.len(),
                );
            }
        });
        response.insert("mem_allocate_time", mem_allocate_time);

        // Step 3: model evaluate in Python
        let python_compute_time = util::record_time(|_| {
            let mut eva_task_map = HashMap::new();
            eva_task_map.insert("config_file", self.config_file.clone());
            eva_task_map.insert("spi_seconds", data_query_time.to_string());
            eva_task_map.insert("rows", self.batch_size.to_string());

            let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

            run_python_function(
                &PY_MODULE_INFERENCE,
                &eva_task_json,
                "model_inference_compute_shared_memory_write_once_int",
            );
        });
        response.insert("python_compute_time", python_compute_time);

        let overall_elapsed_time = util::time_since(&overall_start_time);
        let diff_time =
            model_init_time + data_query_time + python_compute_time - overall_elapsed_time;

        response.insert("overall_query_latency", overall_elapsed_time);
        response.insert("diff", diff_time);

        record_results_py(&response);

        serde_json::json!(response)
    }
}

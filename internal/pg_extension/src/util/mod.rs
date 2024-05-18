use std::time::Instant;

pub fn time_since(start: &Instant) -> f64 {
    Instant::now().duration_since(*start).as_secs_f64()
}

pub fn record_time_returns<F, R>(mut f: F) -> (f64, R)
where
    F: FnMut(&Instant) -> R,
{
    let start_time = Instant::now();

    let result = f(&start_time);

    (time_since(&start_time), result)
}

pub fn record_time<F, R>(f: F) -> f64
where
    F: FnMut(&Instant) -> R,
{
    record_time_returns(f).0
}

pub fn record_time_move<F, R>(f: F) -> f64
where
    F: FnOnce(&Instant) -> R,
{
    let start_time = Instant::now();

    f(&start_time);

    time_since(&start_time)
}

# v2.35.3 code path check (extended_window=False)

Dataset: `realistic_pv` (~4500 W peak).

|backend|mean@03|mean@12|mean@18|peak_hour|note|
|---|---:|---:|---:|---:|---|
|nlinear|19.4|1629.1|-59.9|12|ok at 03:00|
|sparsetsf|86.6|1599.8|-37.4|12|small 3 AM bias|
|lstm|35.2|765.4|433.4|13|small 3 AM bias|
|cnn|68.7|853.0|348.9|11|small 3 AM bias|

|model_size   | Mode             |   Context Length |   Avg Time (ms) |   Avg Forward (ms) |   Avg Backward (ms) |   Avg Loss (ms) |   Avg Optimizer (ms) |
|:------------|:-----------------|-----------------:|----------------:|-------------------:|--------------------:|----------------:|---------------------:|
| small       | forward_backward |              128 |         63.5739 |            23.5363 |             28.8362 |        0.118901 |              11.0826 |
| small       | forward_backward |              256 |         93.0015 |            27.2838 |             54.9447 |        0.173165 |              10.5999 |
| small       | forward_backward |              512 |        174.54   |            54.8532 |            108.813  |        0.328279 |              10.5455 |
| small       | forward_backward |             1024 |        386.958  |           126.523  |            248.748  |        0.575006 |              11.1128 |
| medium      | forward_backward |              128 |         176.827 |            67.4514 |             88.2531 |        0.127479 |              20.9951 |
| medium      | forward_backward |              256 |         274.002 |            86.3308 |            166.33   |        0.267689 |              21.073  |
| medium      | forward_backward |              512 |         534.105 |           173.838  |            338.649  |        0.464218 |              21.1539 |
| medium      | forward_backward |             1024 |        1217.76  |           411.522  |            784.824  |        0.580027 |              20.836  |
| large       | forward_backward |              128 |         343.252 |            104.267 |             194.399 |        0.245941 |              44.3397 |
| large       | forward_backward |              256 |         594.117 |            187.795 |             361.437 |        0.319405 |              44.5663 |
| large       | forward_backward |              512 |        1131.75  |            366.917 |             720.001 |        0.456886 |              44.3758 |
| large       | forward_backward |             1024 |         nan     |            nan     |             nan     |      nan        |             nan      |
| xl          | forward_backward |              128 |         628.554 |            171.016 |             368.209 |        0.269872 |              89.0588 |
| xl          | forward_backward |              256 |        1171.37  |            361.204 |             720.678 |        0.32098  |              89.1624 |
| xl          | forward_backward |              512 |        2348.15  |            741.95  |            1516.47  |        0.449625 |              89.2817 |
| xl          | forward_backward |             1024 |         nan     |            nan     |             nan     |      nan        |             nan      |
| 2.7B        | forward_backward |              128 |         999.304 |            293.866 |             566.771 |        0.40427  |              138.263 |
| 2.7B        | forward_backward |              256 |        1873.13  |            619.069 |            1115.4   |        0.412358 |              138.245 |
| 2.7B        | forward_backward |              512 |         nan     |            nan     |             nan     |      nan        |              nan     |
| 2.7B        | forward_backward |             1024 |         nan     |            nan     |             nan     |      nan        |              nan     |
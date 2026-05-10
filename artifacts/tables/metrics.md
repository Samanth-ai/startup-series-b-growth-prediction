| split    | spec     | model                  |   threshold |   accuracy |   precision |   recall |       f1 |   roc_auc |   pr_auc |
|:---------|:---------|:-----------------------|------------:|-----------:|------------:|---------:|---------:|----------:|---------:|
| random   | expanded | hist_gradient_boosting |        0.46 |   0.836871 |    0.8117   | 0.727869 | 0.767502 |  0.911196 | 0.884401 |
| random   | expanded | random_forest          |        0.5  |   0.789569 |    0.697744 | 0.760656 | 0.727843 |  0.860143 | 0.811449 |
| random   | expanded | mlp                    |        0.34 |   0.782899 |    0.680516 | 0.778689 | 0.7263   |  0.86507  | 0.814801 |
| random   | expanded | logistic_regression    |        0.58 |   0.779867 |    0.693878 | 0.72459  | 0.708901 |  0.862761 | 0.798123 |
| random   | strict   | hist_gradient_boosting |        0.35 |   0.717404 |    0.590452 | 0.770492 | 0.668563 |  0.813523 | 0.747453 |
| random   | strict   | random_forest          |        0.47 |   0.665252 |    0.531453 | 0.803279 | 0.639687 |  0.764596 | 0.647474 |
| random   | strict   | logistic_regression    |        0.48 |   0.6604   |    0.53125  | 0.696721 | 0.602837 |  0.730601 | 0.611161 |
| random   | strict   | mlp                    |        0.36 |   0.684657 |    0.565217 | 0.639344 | 0.6      |  0.742485 | 0.658474 |
| temporal | expanded | hist_gradient_boosting |        0.36 |   0.691328 |    0.93     | 0.652377 | 0.766835 |  0.825428 | 0.943251 |
| temporal | expanded | logistic_regression    |        0.58 |   0.683445 |    0.9087   | 0.659392 | 0.764228 |  0.791014 | 0.92813  |
| temporal | expanded | random_forest          |        0.58 |   0.598545 |    0.949349 | 0.511302 | 0.66464  |  0.791036 | 0.92536  |
| temporal | expanded | mlp                    |        0.45 |   0.570042 |    0.944272 | 0.475448 | 0.632452 |  0.790269 | 0.928295 |
| temporal | strict   | hist_gradient_boosting |        0.27 |   0.695573 |    0.864613 | 0.721746 | 0.786746 |  0.714199 | 0.881529 |
| temporal | strict   | random_forest          |        0.47 |   0.681625 |    0.878244 | 0.685892 | 0.770241 |  0.715568 | 0.879095 |
| temporal | strict   | logistic_regression    |        0.51 |   0.559733 |    0.845963 | 0.530787 | 0.652299 |  0.635592 | 0.843032 |
| temporal | strict   | mlp                    |        0.26 |   0.554275 |    0.851282 | 0.517537 | 0.643723 |  0.659147 | 0.857953 |
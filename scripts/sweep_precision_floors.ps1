$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

python -m scripts.benchmark_oracle

foreach ($floor in 0.2, 0.4, 0.5, 0.6, 0.75, 0.8) {
    python -m src.train --demo --epochs 30 --minimum-precision $floor --output "models/fraud_mlp_$floor.pt"
}

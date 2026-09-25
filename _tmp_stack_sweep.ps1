$py = ".\.venv\Scripts\python.exe"
$inputs = @(
  "train_strict_heldout_gaussian085_presence",
  "train_triangle_clutter256_presence",
  "train_heldout_finalrank020_presence",
  "train_strict_heldout_fusion085_065_t54"
)
$gains = @(0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
New-Item -ItemType Directory -Path outputs\stack -Force | Out-Null
foreach ($inp in $inputs) {
  foreach ($g in $gains) {
    $tag = "$inp" + "_own" + ($g -replace '\.', 'p')
    $out = "outputs\stack\$tag.csv"
    if (Test-Path $out) { continue }
    & $py ownership_refiner.py --root . `
      --input "outputs\$inp.csv" --output $out --split train `
      --raw-cache outputs\cache_train_adaptive `
      --gaussian-cache outputs\cache_train_gaussian085 `
      --min-gain $g *> $null
    if ($LASTEXITCODE -ne 0) { Write-Output "FAILED $tag" }
  }
}
Write-Output "done"
Get-ChildItem outputs\stack -Filter *.csv | Measure-Object | Select-Object -ExpandProperty Count

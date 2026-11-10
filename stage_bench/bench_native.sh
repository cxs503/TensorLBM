set -e
gcc -O3 -fopenmp -march=native -o les_chan les_chan.c -lm
echo "cpus=$(nproc) gcc=$(gcc -dumpversion)"
for th in 8 32 64 128; do
  s=$(date +%s%N)
  OMP_NUM_THREADS=$th ./les_chan --nx 200 --ny 66 --nz 100 --steps 400 --warmup 100 --sample 50 --re-tau 178.12 --u-tau 0.006 --cs 0.1 --threads $th --out /tmp/perf_$th 2>/dev/null
  e=$(date +%s%N)
  python3 -c "d=($e-$s)/1e9; c=200*66*100*400; print(f'threads=$th elapsed={d:.2f}s Mcs/s={c/d/1e6:.1f}')"
done

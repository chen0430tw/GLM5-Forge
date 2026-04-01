set -e
module load miniconda3/24.11.1 >/dev/null 2>&1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
for H in 8 16 32; do
  echo "==== H_Q:${H} ===="
  GLM5_NUM_HEADS=${H} python /work/twsuday816/glm5_reconstruction_test/test_flash_mla_glm5_mla_remote.py || true
done
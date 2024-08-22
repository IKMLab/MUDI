input_path=""
output_path=""

export CUDA_VISIBLE_DEVICES=3

while [ "$#" -gt 0 ]; do
  case "$1" in
    -i) input_path="$2"; shift 2;;
    -o) output_path="$2"; shift 2;;
    --) shift; break;;
    -*) echo "Error: Unsupported flag $1" >&2; exit 1;;
    *) handle_argument "$1"; shift; break;;
  esac
done

python3 src/compute_scores.py \
    -i "$input_path" \
    -o "$output_path"

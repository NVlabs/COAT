export MODEL_NAME="meta-llama/Llama-2-7b-hf"
export CONVERTED_MODEL_PATH="converted_models/llama-2-7b"
export CURRENT_DIR=$(pwd)
export FULL_CONVERTED_MODEL_PATH="${CURRENT_DIR}/${CONVERTED_MODEL_PATH}"

# Get absolute path of current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"


export COAT_PATH=$(pip show fp8_coat | grep "Editable project location" | awk -F': ' '{print $2}')
echo "COAT package is located at: $COAT_PATH"
if [ -z "$COAT_PATH" ]; then
    export COAT_PATH=$(pip show fp8_coat | grep "Location" | awk -F': ' '{print $2}')
fi

cd $COAT_PATH
python -m coat.models.coat_llama_convert_from_hf \
    --model_name $MODEL_NAME \
    --save_path $FULL_CONVERTED_MODEL_PATH \
    --quantize_model true \
    --fabit E4M3 \
    --fwbit E4M3 \
    --fobit E4M3 \
    --bwbit E5M2 \
    --babit E5M2 \
    --bobit E5M2 \
    --group_size 16
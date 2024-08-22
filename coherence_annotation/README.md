## Coherence Relations Annotation

We annotated coherence relations with four methods:
1. Open Source Large Language Models
    * [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
    * [LLaMA-3-70B-Instruct](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16)

2. GPT-4

3. Zero-shot Classification
    * [bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)
    * [roberta-large-mnli](https://huggingface.co/FacebookAI/roberta-large-mnli)

### Usage
* using LLaMA-3-70B-Instruct:
    ```bash
    python coherence_annotation_using_llm.py \
        -m ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16 \
        -q aqlm \
        -i ../dataset/ConvAI2/train_self_original.txt \
        -o <output_json_file_path> \
        --temperature 0.1 \
        --top_p 0.8 \
        --max_tokens 8192 \
        --sample-size -1 \
        --start-index 0
    ```

* using OpenAI API:
    ```bash
    python coherence_annotation_using_openai_api.py \
    -m gpt-4-turbo \
    -i .../dataset/ConvAI2/train_self_original.txt \
    -o <output_json_file_path> \
    --sample-size -1 \
    --start-index 0
    ```
    * If you want to use the OpenAI API, you need to set the OPENAI_API_KEY environment variable in `.env` file.

* Using Zero-shot Classification:
    ```bash
    python coherence_annotation_using_zero_shot.py \
    -m facebook/bart-large-mnli \
    -i .../dataset/ConvAI2/train_self_original.txt \
    -o <output_json_file_path> \
    --sample-size -1 \
    --start-index 0
    ```

## Keyword (Feature) Extraction
We utilize Open Source Large Language Models [LLaMA-3-70B-Instruct](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16) to extract keywords from the query, dialogue context, and persona descriptions. The extracted keywords are used to compute the `Feature Coverage Ratio (FCR)` metrics.

### Usage
* using LLaMA-3-70B-Instruct:
    ```bash
    python keyword_extraction_using_llm.py \
        -m ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16 \
        -q aqlm \
        -i ../dataset/ConvAI2/train_self_original.txt \
        -o <output_json_file_path> \
        --temperature 0.1 \
        --top_p 0.8 \
        --max_tokens 8192 \
        --sample-size -1 \
        --start-index 0
    ```
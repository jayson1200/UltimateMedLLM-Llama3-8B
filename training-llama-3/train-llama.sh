tune download meta-llama/Meta-Llama-3-8B --output-dir .\checkpoints --hf-token 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
tune run lora_finetune_single_device --config llama3/8B_lora_single_device
# Plot 1
# python run_notebook.py -n patchscope_shifted_translation -b 8 -m Llama-2-7b\
#  --model-path /dlabscratch1/public/llm_weights/llama2_hf/Llama-2-7b-hf

# Plot 2
python run_notebook.py -n obj_patch_translation -b 8 -m Llama-2-7b --model-path /dlabscratch1/public/llm_weights/llama2_hf/Llama-2-7b-hf

# Patchscope lens
python run_notebook.py -n translation_lens -b 8 -m Llama-2-7b --model-path /dlabscratch1/public/llm_weights/llama2_hf/Llama-2-7b-hf
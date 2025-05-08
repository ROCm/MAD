from huggingface_hub import snapshot_download
import transformers
import os

model_name = "mosaicml/mpt-30b"
model_revision = "669d63367d4dcbb829e975d52303afa905f9afea"
hf_home_path = os.getenv('HF_HOME', '/data/llm-foundry')

snapshot_download(repo_id=model_name, resume_download=True, revision=model_revision)

model = transformers.AutoModelForCausalLM.from_pretrained(
  'mosaicml/mpt-30b',
  revision=model_revision,
  trust_remote_code=True
)

print(f"Model {model_name} downloaded successfully")

download_path = os.path.join(hf_home_path, "modules/transformers_modules/mosaicml/mpt-30b/")
modeling_file = os.path.join(download_path, model_revision, "modeling_mpt.py")

if os.path.exists(modeling_file):
    print(f"Editing file: {modeling_file}")

    with open(modeling_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Latest main of Flash Attention introduced bug that breaks MPT30B training, thus we manually fix here.
    # Issue tracked by https://huggingface.co/mosaicml/mpt-30b/discussions/22
    new_content = content.replace('(_, indices_q, cu_seqlens_q, max_seqlen_q) = unpadding_function(', '(_, indices_q, cu_seqlens_q, max_seqlen_q, *rest) = unpadding_function(')
    new_content = new_content.replace('(_, indices_k, cu_seqlens_k, max_seqlen_k) = unpadding_function(', '(_, indices_k, cu_seqlens_k, max_seqlen_k, *rest) = unpadding_function(')
    new_content = new_content.replace('(_, indices_v, _, _) = unpadding_function(', '(_, indices_v, _, _, *rest) = unpadding_function(')

    with open(modeling_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("File successfully modified.")
else:
    print(f"Error: File not found - {modeling_file}")
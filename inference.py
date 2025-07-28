import torch
from unsloth import FastLanguageModel
from snac import SNAC
import scipy.io.wavfile
import os

# Modify model loading to use load_in_4bit=True to reduce memory usage
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "The-Data-Dilemma/Medibeng-Orpheus-3b-0.1-ft",
    max_seq_length= 2048,
    dtype = None,
    load_in_4bit = True, # Changed to True
)

prompts = [
   "আপনার শারীরিক পরীক্ষা করতে হবে, and we will schedule that shortly.",
]
chosen_voice = None

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

FastLanguageModel.for_inference(model)

snac_model.to("cpu")

prompts_ = [(f"{chosen_voice}: " + p) if chosen_voice else p for p in prompts]

all_input_ids = []

for prompt in prompts_:
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids
  all_input_ids.append(input_ids)

start_token = torch.tensor([[ 128259]], dtype=torch.int64)
end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

all_modified_input_ids = []
for input_ids in all_input_ids:
  modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
  all_modified_input_ids.append(modified_input_ids)

all_padded_tensors = []
all_attention_masks = []
max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
for modified_input_ids in all_modified_input_ids:
  padding = max_length - modified_input_ids.shape[1]
  padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
  attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
  all_padded_tensors.append(padded_tensor)
  all_attention_masks.append(attention_mask)

all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
all_attention_masks = torch.cat(all_attention_masks, dim=0)

input_ids = all_padded_tensors.to("cuda")
attention_mask = all_attention_masks.to("cuda")
generated_ids = model.generate(
      input_ids=input_ids,
      attention_mask=attention_mask,
      max_new_tokens=1200,
      do_sample=True,
      temperature=0.6,
      top_p=0.95,
      repetition_penalty=1.1,
      num_return_sequences=1,
      eos_token_id=128258,
     use_cache = True
  )
token_to_find = 128257
token_to_remove = 128258

token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

if len(token_indices[1]) > 0:
    last_occurrence_idx = token_indices[1][-1].item()
    cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
else:
    cropped_tensor = generated_ids

mask = cropped_tensor != token_to_remove

processed_rows = []

for row in cropped_tensor:
    masked_row = row[row != token_to_remove]
    processed_rows.append(masked_row)

code_lists = []

for row in processed_rows:
    row_length = row.size(0)
    new_length = (row_length // 7) * 7
    trimmed_row = row[:new_length]
    trimmed_row = [t - 128266 for t in trimmed_row]
    code_lists.append(trimmed_row)


def redistribute_codes(code_list):
  layer_1 = []
  layer_2 = []
  layer_3 = []
  for i in range((len(code_list)+1)//7):
    layer_1.append(code_list[7*i])
    layer_2.append(code_list[7*i+1]-4096)
    layer_3.append(code_list[7*i+2]-(2*4096))
    layer_3.append(code_list[7*i+3]-(3*4096))
    layer_2.append(code_list[7*i+4]-(4*4096))
    layer_3.append(code_list[7*i+5]-(5*4096))
    layer_3.append(code_list[7*i+6]-(6*4096))
  codes = [torch.tensor(layer_1).unsqueeze(0),
         torch.tensor(layer_2).unsqueeze(0),
         torch.tensor(layer_3).unsqueeze(0)]

  audio_hat = snac_model.decode(codes)
  return audio_hat

my_samples = []
for code_list in code_lists:
  samples = redistribute_codes(code_list)
  my_samples.append(samples)

if len(prompts) != len(my_samples):
  raise Exception("Number of prompts and samples do not match")
else:
  output_dir = "generated_audio"
  os.makedirs(output_dir, exist_ok=True)
  for i in range(len(my_samples)):
    print(f"Prompt: {prompts[i]}")
    samples = my_samples[i]
    output_filename = os.path.join(output_dir, f"output_{i}.wav")
    scipy.io.wavfile.write(output_filename, 24000, samples.detach().squeeze().to("cpu").numpy())
    print(f"Saved audio to {output_filename}")

del my_samples, samples

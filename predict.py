#
# Modified from LLaVA/predict.py
# Please see ACKNOWLEDGEMENTS for details about LICENSE
#
import os
import argparse

import torch
from PIL import Image
import json

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def predict(args):
    # Remove generation config from model folder
    # to read generation parameters from args
    model_path = os.path.expanduser(args.model_path)
    generation_config = None
    if os.path.exists(os.path.join(model_path, 'generation_config.json')):
        generation_config = os.path.join(model_path, '.generation_config.json')
        os.rename(os.path.join(model_path, 'generation_config.json'),
                  generation_config)

    # Load model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device="mps")
    device = torch.device("mps")

    # Construct prompt
    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    # --- INIZIO MODIFICA: Creazione Attention Mask ---
    # Creiamo la maschera confrontando gli input_ids con il pad_token_id.
    # Se l'ID non è padding, vale True (1), altrimenti False (0).
    # .long() converte i booleani in 0 e 1.
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    print("Attention Mask estratta con successo!")
    print(f"Shape: {attention_mask.shape}")
    # ------------------------------------------------

    # Load and preprocess image
    image = Image.open(args.image_file).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_inputs = image_tensor.unsqueeze(0).to(device=device, dtype=model.dtype)

    # ===== ATS (Adaptive Token Sampling) Implementation =====
    pre_encoded_features = None
    attention_maps = None
    
    # Step 1: Get attention maps from the vision encoder (FastViTHD Stage 5)
    if args.top_k_tokens > 0:
        vision_tower = model.get_model().get_vision_tower()
        with torch.inference_mode():
            result = vision_tower.forward_images(
                image_inputs, return_attention_maps=True
            )
            if isinstance(result, tuple):
                vision_features_full, attention_maps = result
            else:
                vision_features_full = result
                attention_maps = None
        
        # Step 2: Process attention maps to get token importance scores
        # attention_maps shape: (B, num_heads, N, N) where N = H*W (spatial tokens)
        if attention_maps is not None:
            # Average across all attention heads: (B, N, N)
            attn_avg = attention_maps.mean(dim=1)  # (B, num_heads, N, N) -> (B, N, N)
            attn_avg = attn_avg[0]  # Remove batch dimension: (N, N)
            
            # Compute importance: average attention received by each token from all other tokens
            # This gives us how "attended to" each token is
            token_importance = attn_avg.mean(dim=0)  # (N,): how much each token is attended to
            
            # Alternative: use attention from first token (if it acts like CLS)
            # token_importance = attn_avg[0]  # Attention from first token to all others
            
            # Step 3: Top-K sampling
            # Get number of tokens to keep (default: keep top 200 or 50% of tokens, whichever is smaller)
            num_tokens = token_importance.shape[0]
            k = min(args.top_k_tokens if hasattr(args, 'top_k_tokens') and args.top_k_tokens > 0 else 200, 
                    int(num_tokens * 0.5))  # Keep at most 50% of tokens
            
            # Get indices of top-k most important tokens
            _, top_k_indices = torch.topk(token_importance, k, dim=0)
            top_k_indices = top_k_indices.sort().values  # Sort to maintain spatial order
            
            print(f"\n=== ATS (Adaptive Token Sampling) ===")
            print(f"Total visual tokens: {num_tokens}")
            print(f"Selected top-{k} tokens ({100*k/num_tokens:.1f}%)")
            print(f"Token importance range: [{token_importance.min().item():.4f}, {token_importance.max().item():.4f}]")
            
            # Step 4: Filter vision features to keep only top-k tokens
            # vision_features_full shape: (B, N, C) or (N, C)
            if vision_features_full.dim() == 3:
                vision_features_filtered = vision_features_full[:, top_k_indices, :]
            elif vision_features_full.dim() == 2:
                vision_features_filtered = vision_features_full[top_k_indices, :]
            else:
                raise RuntimeError(f"Unexpected vision feature shape: {vision_features_full.shape}")
            
            # Store filtered features (before projector) to use in encode_images
            # These will be passed to prepare_inputs_labels_for_multimodal
            pre_encoded_features = vision_features_filtered
            
            # Project filtered features through mm_projector to get final count
            vision_features = model.get_model().mm_projector(vision_features_filtered)
        else:
            print("Warning: Attention maps not available, using all tokens")
            pre_encoded_features = None
    else:
        # ATS disabled
        pre_encoded_features = None
    
    # Compute number of visual tokens (after ATS filtering if enabled)
    if pre_encoded_features is not None:
        # Use filtered features to compute token count
        if pre_encoded_features.dim() == 3:
            visual_token_len = pre_encoded_features.shape[1]
        elif pre_encoded_features.dim() == 2:
            visual_token_len = pre_encoded_features.shape[0]
        else:
            raise RuntimeError(f"Unexpected vision feature shape: {pre_encoded_features.shape}")
    else:
        # Compute from full encoding
        with torch.inference_mode():
            vision_features = model.encode_images(image_inputs)
        if vision_features.dim() == 3:
            visual_token_len = vision_features.shape[1]
        elif vision_features.dim() == 2:
            visual_token_len = vision_features.shape[0]
        else:
            raise RuntimeError(f"Unexpected vision feature shape: {vision_features.shape}")

    # Build a boolean mask aligned with the LLM input sequence (after the image
    # features are inserted) where True corresponds to visual tokens.
    valid_token_mask = attention_mask[0].bool()
    valid_input_ids = input_ids[0][valid_token_mask]
    visual_token_positions = []
    seq_pointer = 0
    for token_id in valid_input_ids.tolist():
        if token_id == IMAGE_TOKEN_INDEX:
            visual_token_positions.extend(range(seq_pointer, seq_pointer + visual_token_len))
            seq_pointer += visual_token_len
        else:
            seq_pointer += 1
    final_seq_len = seq_pointer
    visual_token_mask = torch.zeros(final_seq_len, dtype=torch.bool)
    if visual_token_positions:
        visual_token_mask[visual_token_positions] = True
    print(f"Identified {len(visual_token_positions)} visual tokens in the prompt expansion.")


    # Run inference
    with torch.inference_mode():
        outputs_forward = model(
            input_ids=input_ids,
            attention_mask=attention_mask, # Usiamo la maschera creata sopra
            images=image_inputs,
            image_sizes=[image.size],
            output_hidden_states=True, # Fondamentale per avere i vettori
            output_attentions=True,
            return_dict=True,
            pre_encoded_features=pre_encoded_features if attention_maps is not None else None
        )
        
        # Estraiamo l'ultimo hidden state
        last_hidden_state = outputs_forward.hidden_states[-1]
        
        # Estraiamo il primo token (indice 0) che funge da CLS/BOS
        # [Batch 0, Token 0, Tutte le feature]
        cls_token_vector = last_hidden_state[0, 0, :]
        
        print(f"Vettore CLS estratto. Shape: {cls_token_vector.shape}")
        # ----------------------------------------------

        # Calcoliamo l'importanza dei token visivi aggregando l'attenzione media
        # dell'ultimo layer a partire dal token CLS verso i token visivi.
        if outputs_forward.attentions is not None and visual_token_positions:
            last_layer_attn = outputs_forward.attentions[-1].mean(dim=1)[0]  # seq x seq
            if last_layer_attn.shape[-1] != final_seq_len:
                print("Attenzione: la lunghezza della sequenza non coincide con la maschera visiva.")
            cls_attention_weights = last_layer_attn[0].detach().cpu()
            visual_attention_scores = cls_attention_weights[visual_token_mask].tolist()
            importance_payload = {
                "visual_token_indices": visual_token_positions,
                "visual_token_attention": visual_attention_scores
            }
            print("Importanza dei token visivi (attenzione media dal CLS):")
            payload_text = json.dumps(importance_payload, indent=2)
            print(payload_text[:2000] + ("..." if len(payload_text) > 2000 else ""))
        else:
            print("Attenzione non disponibile o nessun token visivo trovato.")
        
        import time
        start_time = time.time() # Sarebbe da usare in realtà il TTFT ma va bene così (per ora)
        
        output_ids = model.generate(
            input_ids,
            images=image_inputs,
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=256,
            use_cache=True,
            pre_encoded_features=pre_encoded_features if attention_maps is not None else None)
        
        end_time = time.time()
        print(f"\n\nGenerazione completata in {end_time - start_time:.2f} secondi.")

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)

    # Restore generation config
    if generation_config is not None:
        os.rename(generation_config, os.path.join(model_path, 'generation_config.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava-fastvithd_0.5b_stage3")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="img/netflix.jpg", help="location of image file")
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt for VLM.")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k_tokens", type=int, default=200, 
                        help="Number of top visual tokens to keep after ATS (0 = disable ATS)")
    args = parser.parse_args()

    predict(args)

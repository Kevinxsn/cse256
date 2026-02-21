
import matplotlib.pyplot as plt
import torch

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size):
        # 1. Automatically detect the device the model is on
        device = next(self.model.parameters()).device 

        # 2. Encode and pad the sentence
        wordids = self.tokenizer.encode(sentence)
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        
        # 3. Create the tensor and move it to the SAME device as the model
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0).to(device)

        print(f"Input tensor shape: {input_tensor.shape} on device: {device}")

        # 4. Process through model (ensure it returns attn_maps)
        self.model.eval() # Set to eval mode for sanity check
        with torch.no_grad():
            _, attn_maps = self.model(input_tensor)

        print("Number of attention maps:", len(attn_maps))

        # 5. Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            # attn_map shape: (batch, heads, seq, seq)
            # We take the first head of the first batch for visualization
            # and move it back to CPU for numpy/matplotlib
            att_map_to_plot = attn_map[0, 0].detach().cpu().numpy() 

            # Check normalization: probabilities should sum to 1 over rows
            total_prob_over_rows = torch.sum(attn_map[0, 0], dim=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print(f"Layer {j+1} Head 1: Failed normalization test.")
                print("Total probability over rows:", total_prob_over_rows.cpu().numpy())

            # Create heatmap
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map_to_plot, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()  
            fig.colorbar(cax, ax=ax)  
            plt.title(f"Attention Map (Layer {j + 1}, Head 1)")
            
            # Save the plot
            plt.savefig(f"attention_map_layer_{j + 1}.png")
            plt.show()
    
    
    def sanity_check_decoder(self, sentence, block_size):
        # 1. Automatically detect the device the model is on
        device = next(self.model.parameters()).device 

        # 2. Encode and pad the sentence
        wordids = self.tokenizer.encode(sentence)
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        
        # 3. Create the tensor and move it to the SAME device as the model
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0).to(device)

        print(f"Input tensor shape: {input_tensor.shape} on device: {device}")

        # 4. Process through model
        self.model.eval() 
        with torch.no_grad():
            # For Encoder, this returns (output, attn_maps)
            # For Decoder, this returns (loss/logits, attn_maps)
            _, attn_maps = self.model(input_tensor)

        print("Number of attention maps:", len(attn_maps))

        # 5. Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            # HANDLE SHAPE DIFFERENCE:
            # Encoder map is (B, H, L, L). Decoder map is (B, L, L).
            if attn_map.dim() == 4:
                single_map = attn_map[0, 0] # Pick Batch 0, Head 0
                title_suffix = f"Layer {j+1}, Head 1"
            else:
                single_map = attn_map[0]    # Pick Batch 0
                title_suffix = f"Map {j+1}"

            att_map_to_plot = single_map.detach().cpu().numpy()

            # Check normalization: probabilities should sum to 1 over rows
            total_prob_over_rows = torch.sum(single_map, dim=1)
            if torch.any(total_prob_over_rows < 0.95) or torch.any(total_prob_over_rows > 1.05):
                print(f"{title_suffix}: Failed normalization test.")
                # We use a slightly wider tolerance (0.95-1.05) for float precision

            # Create heatmap
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map_to_plot, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()  
            fig.colorbar(cax, ax=ax)  
            plt.title(f"Attention Map ({title_suffix})")
            
            # Save the plot
            plt.savefig(f"attention_map_{j + 1}.png")
            plt.close() # Close to save memory during long loops


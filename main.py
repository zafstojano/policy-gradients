import torch 
import torch.nn.functional as F

def main():
    a = torch.randn(6)
    b = torch.randn(8)
    how = "right"
    tensors = [a, b]

    max_len = max(t.size(0) for t in tensors)
    padded_tensors = []
    for tensor in tensors:
        pad_len = max_len - tensor.size(0)
        padding = (pad_len, 0) if how == "left" else (0, pad_len)
        padded_tensor = F.pad(tensor, padding)
        print(padded_tensor.shape, padded_tensor)
        padded_tensors.append(padded_tensor)



if __name__ == "__main__":
    main()

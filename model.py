import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, input_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim // 2)

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()

        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        attn_weights = F.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, input_dim)
        output = self.out_proj(attn_output)

        return output

class DFL_MHC(nn.Module):
    # dropout = 0.9
    def __init__(self, input_dim=224, d_model=128, num_heads=4, num_layers=2, num_classes=2, dropout_rate = 0.9):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm1 = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.recurrent_dropout = nn.Dropout(dropout_rate)
        self.attention = MultiHeadSelfAttention(256)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.SELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x,_ = self.lstm1(x)
        x = x.unsqueeze(1)
        x = self.recurrent_dropout(x)
        x = self.attention(x).squeeze(1) # 128

        return self.classifier(x)

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DFL_MHC(
        input_dim=224,
        d_model=128,
        num_heads=4,
        num_layers=2,
        num_classes=2,
        dropout_rate=0.1
    ).to(device)

    model.eval()


    batch_size = 8
    input_dim = 224

    x = torch.randn(batch_size, input_dim).to(device)

    print("Input shape:", x.shape)

    # forward
    with torch.no_grad():
        out = model(x)

    print("Output shape:", out.shape)
    print("Output:", out)



if __name__ == "__main__":
    test()
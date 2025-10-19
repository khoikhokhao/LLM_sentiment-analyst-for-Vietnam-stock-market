# 🇻🇳 PhoBERT-large v2 — Vietnamese Headline Sentiment Classification

## 🧩 Giới thiệu
Đây là mô hình **PhoBERT-large v2**, fine-tune từ `vinai/phobert-large` để **phân loại cảm xúc (pos/neg)** cho **tiêu đề tin tức tiếng Việt**, đặc biệt trong lĩnh vực tài chính – chứng khoán.  
Mục tiêu: xác định sắc thái **tích cực** hoặc **tiêu cực** trong tiêu đề tin nhằm hỗ trợ các hệ thống hiểu cảm xúc văn bản tiếng Việt.

Mô hình được huấn luyện bởi **Phạm Minh Khôi (FPT University, AI major)**.  
Phiên bản v2 được tiếp tục fine-tune từ v1, với trọng số lớp `neg` được tăng (3.5 → 0.5) để cải thiện độ nhạy đối với tiêu đề tiêu cực.

---

## 🏗️ Kiến trúc mô hình

```
PhoBERT-large backbone (24 layers, 16 heads, hidden_size=1024)
        ↓
Vector token <s> (CLS-style)
        ↓
Classification Head:
    Dropout → Dense(1024→1024) + Tanh → Dropout → OutProj(→2 logits)
        ↓
Softmax → [P(neg), P(pos)]
```

- **Base model:** `vinai/phobert-large`
- **Embedding:** 1024-chiều
- **Encoder:** 24 tầng Transformer, 16 đầu attention mỗi tầng
- **Head:** 2 lớp tuyến tính (Dense + Tanh)
- **Loss:** CrossEntropyLoss(weight = [3.5, 0.5])
- **Optimizer:** AdamW + warmup + weight decay
- **Early stopping:** dừng nếu 2 epoch không cải thiện
- **Tokenizer:** PhoBERT BPE (`use_fast=False`)

---

## 📊 Kết quả đánh giá

| Bộ dữ liệu | Accuracy | F1_macro | Precision_macro | Recall_macro |
|-------------|-----------|-----------|-----------------|---------------|
| Validation  | **0.925** | **0.886** | 0.906 | 0.869 |
| Test        | **0.904** | **0.861** | 0.856 | 0.866 |

Ví dụ dự đoán:

| Tiêu đề | Logits [neg,pos] | Kết quả |
|----------|------------------|----------|
| “FPT báo lãi quý 3 tăng 25%, vượt kế hoạch năm” | [-1.2, 1.8] | POS |
| “Doanh nghiệp X lỗ quý 3, hoãn cổ tức” | [1.4, -0.7] | NEG |

---

## 💡 Cách sử dụng

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "phamminhkhoi/phobert-large-v2-sentiment"  # hoặc đường dẫn local
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

text = "Thị trường chứng khoán bật tăng mạnh phiên sáng nay"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
with torch.no_grad():
    logits = model(**inputs).logits
pred = torch.argmax(logits, dim=-1).item()
label = "pos" if pred == 1 else "neg"
print(f"Kết quả: {label}")
```

---

## 👨‍💻 Tác giả
**Phạm Minh Khôi**  
FPT University – AI Major  
📧 Contact: [phamminhkhoi.05.09.12@gmail.com]  
📦 HuggingFace: * https://huggingface.co/khoidan/phobert-large-v2-sentiment_stock_Vietnam_headlinepaper *

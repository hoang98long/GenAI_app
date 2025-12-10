# VietGenAI RAG App

Một dự án mẫu xây dựng sản phẩm GenAI cho tiếng Việt, sử dụng:
- **LLM**: [`AITeamVN/GRPO-VI-Qwen2-7B-RAG`](https://huggingface.co/AITeamVN/GRPO-VI-Qwen2-7B-RAG) – tối ưu cho hội thoại & RAG tiếng Việt.
- **Embedding**: [`hiieu/halong_embedding`](https://huggingface.co/hiieu/halong_embedding) – sentence-transformers tối ưu cho RAG tiếng Việt.

Ứng dụng cung cấp API (FastAPI) cho phép:
- Upload tài liệu (PDF, DOCX, TXT…)
- Tạo vector store từ tài liệu bằng embedding tiếng Việt
- Hỏi đáp nhiều lượt dựa trên ngữ cảnh đã index (RAG)
- Ghi log, theo dõi chi phí và chất lượng cơ bản

---

## Kiến trúc chính

- `app/`
  - `api/`: FastAPI, route `/health`, `/chat`, `/ingest`
  - `workers/`: worker để ingest tài liệu, rebuild index
  - `cli/`: lệnh dòng lệnh cho quản trị (ingest thủ công, test…)

- `llm/`
  - `models/`: client cho LLM (Qwen2-Vi RAG, hoặc model khác), router đa model
  - `retrieval/`: loader PDF/DOCX, chunking, embedding, vector store
  - `pipelines/`: pipeline RAG (retrieval-augmented generation)
  - `prompts/`: prompt templates tiếng Việt
  - `evaluation/`: script đánh giá chất lượng QA
  - `monitoring/`: logging, thu feedback người dùng

- `data/`
  - `raw/`: tài liệu gốc (read-only)
  - `processed/`: text đã chuẩn hoá & chunk
  - `vectorstores/`: index nhúng
  - `eval_sets/`: bộ câu hỏi đánh giá

- `tests/`: unit, integration & eval regression
- `notebooks/`: thử nghiệm, phân tích
- `scripts/`: ingest & quản lý dữ liệu
- `docker/`, `infra/`: container & deploy (tuỳ chọn)

---

## Yêu cầu hệ thống

- Python 3.10+
- GPU (khuyến nghị) với CUDA nếu muốn chạy LLM local nhanh.
  - Nếu không có GPU có thể:
    - Dùng bản đã quantize (gguf) qua thư viện khác, **hoặc**
    - Gọi LLM qua API (cấu hình lại `LLM_PROVIDER` trong `.env`).

---

## Cài đặt nhanh (dev)

```bash
# 1. Tạo môi trường
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Cài đặt phụ thuộc
pip install -U pip
pip install .                  # dùng pyproject.toml

# 3. Tạo file .env từ mẫu
cp .env.example .env

# 4. Chạy API
uvicorn app.api.main:app --reload

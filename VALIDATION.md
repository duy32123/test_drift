# Phần đã kiểm và phần còn phải chạy trên máy đích

## Môi trường kiểm tại thời điểm đóng gói

- Python 3.12.14, PyTorch 2.14.0+cpu, torchvision 0.29.0+cpu.
- transformers 4.57.6. Không có CUDA/GPU trong môi trường kiểm.
- Xem `TEST_RESULTS.txt` cho output pytest của đúng bản đóng gói.

## Đã thực thi

- Allocator: 25 bài PSD ngẫu nhiên đối chiếu objective và feasibility với solver
  SLSQP độc lập; negative budget/recovery, infeasible candidate, signed objective,
  cross-factor terms và LR scaling.
- Bước cập nhật: NS5 được tái dựng bằng spectral proposal, zero momentum không
  dịch chuyển, gradient coordinates đúng, rollback bit-exact, finite LoRA
  cross-term và lỗi batch-mean Fisher được kiểm bằng ví dụ tính trực tiếp.
- Pipeline ba task tổng hợp chạy hết với cả năm nhánh; resume tại ranh giới task
  cho cùng trọng số và metrics với chạy liên tục. Kiểm memory anchors và split.
- Đường code Hugging Face CLIP được kiểm bằng CLIP nhỏ khởi tạo ngẫu nhiên,
  tokenizer/pretrained loader được mock: injection đủ q/k/v/out/fc1/fc2,
  gradient checkpointing, backward và autograd.grad hoạt động. Không tải model
  pretrained; đây không phải kết quả CLIP ViT-B/16.
- `check_env.py`: CPU matmul/backward/NS5/SVD/restore trên factor shapes 16×768,
  768×16, 16×3072 và 3072×16.
- Checkpoint pretrained `openai/clip-vit-base-patch16` chính thức đã được tải và
  nạp thành công trên CPU (149,620,737 tham số, revision
  `57c216476eefef5ab752ec549e440a49ae4ae5f3`). Chưa chạy forward của model đó.
- Đường transform/split CIFAR thật được kiểm bằng dataset giả lập đúng topology
  100 lớp. Endpoint CIFAR truy cập được nhưng download thật bị dừng ở 39.7/169 MB
  để bàn giao ngay; archive dữ liệu chưa hoàn chỉnh không nằm trong gói mã.
- Các lệnh `train.py`, `evaluate.py`, `summarize.py`, `profile_batches.py` chạy
  thành công trên fixture tổng hợp; profile hai micro-batch dừng đúng số update.
- `run_pilot.py` được kiểm skip run hoàn tất, mở rộng từ một lên hai task bằng
  resume, từ chối config đã đổi và sinh bảng tổng hợp.

## Chưa kiểm

- CUDA/Blackwell, BF16 trên RTX 5050, cuSOLVER và peak VRAM thực tế.
- Download/preprocessing của CIFAR thật và inference/training pretrained B/16.
- Hai task CIFAR, đủ 10 task, throughput và overhead Fisher/gate trên GPU.
- Chất lượng method, frontier sau tuning, seed replication, hoặc khớp số paper.

Một test CUDA được skip nếu không có GPU. Khi chạy lại trên card của bạn nó phải
thực sự chạy, cùng với `check_env.py --require-cuda`. Các test offline kiểm độ
đúng của code trong phạm vi trên, không chứng minh method cải thiện CL.

## Thứ tự chạy tiếp

1. Cài wheel CUDA phù hợp, chạy environment check và test suite.
2. Profile task đầu để chọn micro-batch; sau đó hai task để đo overhead đầy đủ.
3. So Muon, scalar, replay và drift trên validation; tune LR riêng, ghi tài
   nguyên. Tăng Fisher samples nếu proxy không ổn, không mặc định 8 là đủ.
4. Khóa config rồi chạy thêm seed và đánh giá test. Chỉ so số paper khi các
   khác biệt protocol đã được giải quyết hoặc được công bố rõ ràng.

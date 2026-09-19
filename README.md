# Drift-CIL v0.2 — bộ chạy nghiên cứu cho RTX 5050 8 GB

LoRA4CIL trên CLIP ViT-B/16 + CIFAR-100, 10 task, backbone và text head đóng băng.
Đây là **research reimplementation**, chưa phải tái lập chính xác Table 7.
Các pilot CIFAR/GPU nằm trong `runs/`; Gate 1 cũ chưa dùng checkpoint task A chung.
Các kiểm tra offline không phải bằng chứng method thắng.

## Những gì đã có

- Sáu nhánh: `adamw`, `muon`, `scalar`, `replay`, `drift`, `drift_scalar`.
- Muon NS5 và các nhánh dùng cùng EMA/Nesterov, normalization, shape scaling,
  dtype FP32 của NS và số vòng lặp. Có backend `polar` cho đối chiếu riêng.
- Thin SVD trên **bước factor thực sự**, đã gồm LR, NS amplitudes và scaling.
  Phân bổ không nâng singular values nhỏ thành 1. Ma trận momentum bằng 0 không di chuyển.
- Ngân sách loss toàn cục; Fisher ghép các factor, giữ cross-factor terms.
  Solver dùng score matrix N×d và matvec, không dựng Fisher d×d đặc.
- Loss cũ có dấu; ngân sách neo theo loss lúc mẫu được thêm vào memory.
- Đo lại loss thật, backtracking và khôi phục snapshot chính xác.
- Gradient accumulation, validation riêng, CIL không dùng task-ID lúc đánh giá,
  checkpoint theo task, resume và JSONL chi phí/vi phạm thực tế.

**Giới hạn:** ngân sách là mean cross-entropy trên memory, không bảo đảm accuracy
hay từng task; có thể overfit memory. Fisher vẫn là proxy độ cong. Thuật toán có
thể vượt ngân sách trước bước vì memory được thay đổi; bước recovery giảm loss
nhưng chưa đạt trần được ghi riêng, không báo là feasible.

## 1. Cài môi trường trên máy của bạn

Dùng Python 3.11 hoặc 3.12. Các lệnh dưới dùng Windows PowerShell và không cần
thay execution policy. Chạy trong thư mục đã giải nén:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_env.py --require-cuda
.\.venv\Scripts\python.exe test_drift_muon.py
```

Linux: tạo venv bằng `python3 -m venv .venv`, dùng `.venv/bin/python` thay đường
dẫn Windows. Chọn wheel PyTorch hỗ trợ Blackwell/CUDA 12.8 hoặc mới hơn theo
https://pytorch.org/get-started/locally/ và driver tương ứng. Không dùng bộ
PyTorch 2.1/CUDA 11.8 của repo MoE cũ.

`check_env.py` thực sự chạy matmul, backward, NS5, SVD và restore trên CUDA.
Chỉ thấy `(12, 0)` chưa đủ chứng minh các kernel hoạt động. Test CUDA sẽ được
skip khi không có GPU; CPU pass không thay thế test trên card của bạn.

Lưu môi trường thành công để tái lập:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > environment-lock.txt
```

## 2. Kiểm pipeline offline, không tải dữ liệu/model

```powershell
.\.venv\Scripts\python.exe train.py --config configs/smoke.yaml --method drift --output runs/smoke
```

Tiny model + dữ liệu sinh chỉ kiểm luồng train/memory/checkpoint. Không báo những
accuracy này trong bảng benchmark. Các test tích hợp chạy cả sáu nhánh và kiểm
resume cho kết quả giống hệt một run không ngắt.

## 3. Profile CLIP thật trước khi chọn micro-batch

Kiểm toàn bộ đường chạy CIFAR thật bằng một ảnh/class và hai task trước. Lệnh
này đi qua backward, memory, Fisher, allocator, acceptance và checkpoint nhưng
gắn nhãn `debug_*`; số accuracy của nó không phải benchmark:

```powershell
.\.venv\Scripts\python.exe preflight_cifar.py --download --require-cuda --output runs/cifar_preflight
```

Sau khi model và dữ liệu đã có trong cache, có thể dùng `--offline` thay
`--download`. Nếu download bị ngắt, xóa file tar CIFAR chưa hoàn chỉnh rồi chạy
lại; torchvision không tiếp tục từ phần tar dở.

```powershell
.\.venv\Scripts\python.exe profile_batches.py --config configs/cifar100.yaml --batches 4 8 16 --updates 10 --output runs/profiles --download
```

Lần đầu cần mạng để tải CIFAR-100 và checkpoint `openai/clip-vit-base-patch16`.
Không đóng gói dataset/trọng số trong ZIP. Có thể đặt `data_root` đến CIFAR đã
tải và dùng cache Hugging Face sẵn có. Sau đó bỏ `--download`.

Mỗi profile là process mới và chỉ đo task đầu; **chưa đo overhead memory/QP**.
Xem `profiles.json`, từng log và peak VRAM. Giữ effective batch 128. Nếu OOM,
hạ `micro_batch`, `memory_micro_batch` và `eval_batch` (các giá trị sau nằm trong
YAML). Tăng accumulation không làm biến mất chi phí Fisher/acceptance ở mỗi update.

## 4. Hai task đầu trên benchmark thật

Sau khi chọn micro-batch, sửa YAML hoặc truyền `--micro-batch`. Chạy mỗi nhánh
vào thư mục riêng. Ví dụ dưới là hai task đầy đủ theo số epoch trong YAML:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/cifar100.yaml --method muon --task-limit 2 --output runs/pilot_muon --micro-batch 8
.\.venv\Scripts\python.exe train.py --config configs/cifar100.yaml --method scalar --task-limit 2 --output runs/pilot_scalar --micro-batch 8
.\.venv\Scripts\python.exe train.py --config configs/cifar100.yaml --method replay --task-limit 2 --output runs/pilot_replay --micro-batch 8
.\.venv\Scripts\python.exe train.py --config configs/cifar100.yaml --method drift --task-limit 2 --output runs/pilot_drift --micro-batch 8
.\.venv\Scripts\python.exe summarize.py runs/pilot_muon runs/pilot_scalar runs/pilot_replay runs/pilot_drift --output comparison.csv
```

Hoặc dùng hàng đợi tuần tự, tự bỏ qua run hoàn tất và tiếp tục từ checkpoint
task gần nhất khi chạy lại cùng lệnh:

```powershell
.\.venv\Scripts\python.exe run_pilot.py --config configs/cifar100.yaml --output runs/pilot --methods muon scalar replay drift --seeds 1993 --task-limit 2
```

`--method-lr drift=0.0001` cho phép đặt LR riêng. Hàng đợi từ chối tái sử dụng
thư mục nếu config đã đổi, ghi `progress.json`, `console.log` và cuối cùng tạo
`comparison.csv`. Dùng `--dry-run` để xem danh sách trước khi chạy.

Dùng `--max-updates 40` để dừng sớm khi debug. Run dừng giữa task được gắn
`profile_stopped`, không coi là benchmark hoàn tất. Schedule dựa trên cả luồng
10 task; `--task-limit` không thay đổi LR của cùng prefix.

Để profile đúng một epoch task đầu, dùng `--epochs 1 --task-limit 1` và sửa
`warmup_steps` cho nhỏ hơn tổng update của cấu hình đó; không diễn giải learning
curve của profile như cấu hình chính.

## 5. Tune, khóa cấu hình, rồi chạy thêm seed

Để so `drift` với scalar restriction `drift_scalar`, dùng checkpoint task A chung:

```powershell
.\.venv\Scripts\python.exe run_paired.py --config configs/smoke.yaml --output runs/paired_smoke
.\.venv\Scripts\python.exe run_paired.py --config configs/cifar100.yaml --output runs/gate1_paired
```

Mỗi lệnh cần thư mục mới. Script train task A một lần, fork hai nhánh đến hết task B,
giữ toàn bộ state checkpoint (momentum, memory/anchors và RNG), cùng lịch LR/config.
`run_pilot.py` vẫn là hàng đợi các run độc lập. `scalar` chỉ backtrack bước cơ sở;
`drift_scalar` giải cùng bài toán ngân sách với `drift`, giới hạn `z = c*1`.

`run_paired.py` lưu `source/`, `provenance.json`, `checksums.json`, checkpoint chung,
step logs, manifest, summary, comparison và audit. Log mỗi nhánh chỉ gồm task B;
log task A nằm trong `task_a/`. Summary hai nhánh bao gồm cùng metrics task A;
thời gian riêng mỗi nhánh không bao gồm chi phí train task A chung.
Giữ toàn bộ thư mục làm artifact để người khác tái audit; checkpoint/source snapshot
không tự động được đưa vào Git. Checksum xác minh artifact, không chứng minh chất lượng method.
`--fork-from` chỉ nhận checkpoint task A mới có source hash LF khớp; resume thường
vẫn yêu cầu cùng method. Chỉ nạp checkpoint tin cậy.

Hash module mới có trường `code_sha256_lf` chuẩn hóa CRLF → LF để đối chiếu Git.
Xem `GATE1_PROVENANCE.md` về hash và giới hạn của hai run Gate 1 cũ.

- Train chỉ dùng training split. Mặc định tách 50 ảnh/class làm validation,
  không lấy memory/Fisher/acceptance từ validation hoặc test.
- LR, delta, damping, số Fisher sample và replay weight đều là hyperparameter.
  Tune trên validation với ngân sách được công bố; `lr=1e-4` không được mặc định
  là optimum của mọi nhánh. AdamW có thể khởi đầu bằng `--lr 1e-5`.
- Các nhánh scalar/replay/drift có cùng capacity memory, nhưng không tự động có
  cùng compute hay số lượt đọc. `comparison.csv` tách các loại old-data access.
  Điều chỉnh `replay_samples` theo ngân sách muốn đối chiếu và báo cáo wall-clock.
- Khóa config rồi chạy `--seed 1993`, `1994`, `1995` vào thư mục khác nhau.
  `class_order_seed` độc lập với seed huấn luyện để ghép cặp task order.
- Bỏ `--task-limit` để chạy đủ 10 task. Giữ ma trận accuracy theo task, không
  chỉ final average. `Avg`, `Last`, `BWT` trong file là số 0–1, nhân 100 để ra %.

Chỉ sau khi khóa config mới đánh giá test của các checkpoint:

```powershell
.\.venv\Scripts\python.exe evaluate.py --run runs/pilot_drift --split test --batch-size 8
```

## Resume và các file đầu ra

```powershell
.\.venv\Scripts\python.exe train.py --config configs/cifar100.yaml --method drift --output runs/pilot_drift --resume runs/pilot_drift/last.pt --task-limit 2
```

Resume tại **ranh giới task đã hoàn tất**, có adapter, momentum/optimizer,
memory, anchors và RNG; một task bị ngắt sẽ chạy lại từ đầu. Không có resume
giữa epoch. Hãy giữ nguyên hyperparameters. Không sửa checkpoint nhận từ nguồn
không tin cậy: `torch.load` ở đây dành cho checkpoint do chính bộ chạy tạo.

- `manifest.json`: toàn bộ config, phiên bản, revision model, class order, shapes.
- `steps.jsonl`: LR, số ảnh mới/cũ đã đọc, timing, trạng thái solver và actual loss gate.
- `summary.json`: validation metrics, trạng thái run và peak GPU memory.
- `task_XX.pt`, `last.pt`: adapter checkpoints, không sao chép backbone lớn.
- `test_evaluation.json`: test metrics do lệnh evaluate riêng tạo ra.

Mã người dùng gửi được giữ nguyên trong `reference/`; **không** phải mã được
import bởi training harness mới. Xem `CHANGES.md` về những thay đổi có chủ đích.
Xem `METHOD.md` cho công thức, dấu, đơn vị bước và quy tắc nhận/reject update.
Xem `VALIDATION.md` để biết chính xác phần đã/chưa kiểm và `PROTOCOL.md` trước
khi so số với paper.

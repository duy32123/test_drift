# Chẩn đoán scalar v0.2 — bản bổ sung, không thay method

## Vì sao tạm dừng

Log cho thấy scalar thử 9 mức từ 1 tới 1/256 và từ chối liên tục.
Đây chưa chứng minh baseline không hợp lệ hoặc cần tăng delta. Nó có thể là
hành vi thật tại biên ngân sách, hoặc do lưới co bước chưa đủ nhỏ. Các spike
84–116 giây chưa đủ để kết luận throttling/hỏng GPU.

Bản này chỉ thu thập bằng chứng. Không đổi delta, tolerance, backend, LR,
độ chính xác FP32 của gate, kích thước memory hay code training gốc.

## Cài và chạy

1. Dừng hàng đợi cũ bằng Ctrl+C, đợi dấu nhắc PowerShell trở lại.
2. Giải nén ba file của gói này vào thư mục drift-cil-v0.2, cạnh train.py.
3. Từ thư mục đó chạy:

```powershell
.\.venv\Scripts\python.exe -m unittest test_diagnose_scalar.py
.\.venv\Scripts\python.exe diagnose_scalar.py --run runs/pilot/scalar_s1993 --output runs/diagnose_scalar --max-updates 60
```

Output phải là thư mục MỚI. Nếu đã chạy chẩn đoán một lần thì dùng
`--output runs/diagnose_scalar_2`; không xóa run gốc.

Script đọc last.pt của SCALAR, tự lấy config và trạng thái ngẫu nhiên từ đó.
Task đã hoàn thành không chạy lại. Phần task đang dở phải tái chạy vì v0.2
không lưu checkpoint giữa task. Không thể khôi phục trọng số đang dở từ log.

Nó chạy tối đa 60 update MỚI, hoặc dừng ngay tại lần reject đầu tiên.
Khi gặp reject, nó lưu snapshot chẩn đoán, kiểm lại loss tại trạng thái cũ,
rồi thử bước 1/512, 1/1024, ... tối đa 1/65536 với cấu hình mặc định.
Mọi bước bổ sung đều được hoàn tác; không được dùng để tiếp tục train.
Thử dừng sớm khi tìm được bước khác 0 chấp nhận được, hoặc khi FP32 làm
bước thành 0. Có thể không tái hiện đúng update 393 do môi trường/CUDA.

Với tốc độ 24–30 giây/update trong log, trần 60 update riêng phần train
khoảng 24–30 phút, cộng nạp model và các phép đo thêm. Thường dừng trước
trần nếu gặp reject sớm. Đây là ước tính có điều kiện, không phải profile.

Không chạy đồng thời với một tiến trình train khác trên GPU 8 GB.

## Gửi lại kết quả

Gửi `runs/diagnose_scalar/diagnostic.json` và `gate_trials.jsonl`.

- `smaller_nonzero_step_is_feasible`: ít nhất một mức dưới lưới cũ hợp lệ,
  nhưng chưa chứng minh bước nhỏ đó mang lại tiến bộ đáng kể trên task mới.
- `smaller_step_only_recovers_toward_budget`: bước giảm loss nhưng chưa đạt trần.
- `reached_parameter_rounding_limit`: bước thử đã thành 0 trong tham số.
- `no_acceptable_step_in_additional_grid`: chỉ nói về các mức đã thử,
  không phải chứng minh mọi bước nhỏ hơn đều bất khả thi.
- `baseline_repeatability_warning`: cần kiểm độ lặp lại trước khi diễn giải
  những chênh lệch sát tolerance.
- `no_rejection_observed_within_limit`: chưa tái hiện trong giới hạn đã chạy.

`diagnostic_state.pt` là snapshot trọng số và hướng tại lần reject để phân tích;
KHÔNG phải checkpoint resume. Không đưa thư mục diagnostic vào bảng benchmark.
Các update trong steps.jsonl không bao gồm update bị dừng trong gate;
update đó được ghi riêng trong diagnostic.json và gate_trials.jsonl.

## Chạy nhánh tiếp theo

Sau khi chẩn đoán kết thúc, có thể chạy replay trong lúc phân tích kết quả:

```powershell
.\.venv\Scripts\python.exe run_pilot.py --config configs/cifar100.yaml --output runs/pilot --methods replay --seeds 1993 --task-limit 2
```

Lệnh này chỉ chọn replay; không khởi động lại scalar. Run Muon và scalar cũ
vẫn giữ nguyên. progress.json và comparison.csv cấp hàng đợi lúc này chỉ phản
ánh danh sách replay được chọn, không đại diện đủ bốn nhánh.

Chưa chạy một sweep drift dài trước khi hiểu gate. Nếu sau này thay delta hoặc
lưới tìm bước, áp dụng quy tắc tương ứng cho cả scalar và drift, dùng output mới
và ghi rõ protocol. Không tăng ngân sách riêng để cứu một nhánh có kết quả xấu.

## Kiểm chứng và giới hạn

7 kiểm tra logic CPU: đối chiếu 1.000 bài với accept_step gốc, acceptance/
recovery/rejection, tìm bước nhỏ, phân biệt bước 0, hoàn tác khi lỗi/Ctrl+C,
và loss không hữu hạn. Các test trích đúng accept_step từ code v0.2 tại máy chạy.

Môi trường tạo gói không có PyTorch/GPU: chưa chạy tích hợp CLIP/CIFAR của addon.
Test logic không phải chứng nhận CUDA, throughput hay chất lượng phương pháp.

Nguồn: bổ sung cho drift-cil-v0.2-handoff(1).zip, không thay thế ZIP gốc.

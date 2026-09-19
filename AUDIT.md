# Audit no-op và nhánh scalar đúng — vá vào code hiện có

Gói trước của tôi là một module rời, không dùng được với repo này. Bản này là
patch vào chính `drift_cil/`, theo đúng interface đang chạy.

## Vấn đề

`accept_step` chỉ hỏi một câu: loss memory ĐO ĐƯỢC có nằm dưới trần không.
Một allocation bằng 0 vượt qua câu hỏi đó miễn phí — `Transaction.apply` copy
lại snapshot rồi cộng `-1.0 * 0`, nên phép đo trùng bit với phép đo trước bước,
và mọi trần mà snapshot đã thỏa thì vẫn thỏa. Update đó được ghi là
`accept_status='accepted'`, `accepted_scale=1.0`, `backtracks=0`.

`steps.jsonl` không ghi `z`. Nên với log đã có, không có cách nào phân biệt một
update thật với một update không làm gì.

Bốn đường dẫn tới `z = 0`, tất cả đều đã tái hiện trên chính
`drift_cil.allocator.solve_budget` (xem `tests/test_audit.py`):

| | điều kiện | status solver báo |
|---|---|---|
| W1 | `rho < 0`, không có hướng phục hồi bậc nhất | `infeasible_candidate` |
| W2 | mọi `b_i <= 0` với `z_lower: 0.0` — nghiệm hộp ĐÚNG là 0 | `unconstrained` |
| W3 | bisection nhắm `rho` nghiêm ngặt còn gate thực thi `ceiling + atol` | `active` |
| W4 | `infeasible_candidate` KHÔNG kéo theo `z = 0` | `infeasible_candidate` |

W4 là lý do phải đọc `||z||`, không đọc chuỗi status: khi `cost(minimum) <= 0`
thì bước least-harm khác 0 được giữ lại.

## Đã sửa gì

**`drift_cil/audit.py` (mới).** Tính lại mỗi update đã làm gì, từ trường thô.
Không bao giờ tin `accept_status`. Trả về cả bằng chứng đằng sau mỗi kết luận,
nên chỗ bất đồng là chỗ kiểm tra được, không phải ý kiến thứ hai. Không cần torch.

**`drift_cil/optim.py`.** `allocate` nhận `tolerance` và ghi
`z_inf`, `z_l1`, `z_nonzero` vào diagnostics — runner log chúng miễn phí.
Thêm `allocate_scalar`: cùng ngân sách, thu về một biên độ toàn cục `z = c*1`.

**`drift_cil/allocator.py`.** Bisection dùng `rho + tolerance` thay vì `rho`
nghiêm ngặt. Trước đó kiểm tra feasibility đầu hàm, guard cuối hàm và
`accept_step` đều dùng `rho + tolerance`, chỉ riêng bisection thì không.

**`drift_cil/runner.py`.** Một `gate_tolerance` duy nhất đi vào cả solver và
gate. Ghi `outcome` đã tính lại vào mỗi record. Thêm method `drift_scalar`.

**`audit_steps.py` (mới).** Chạy trên `steps.jsonl` đã có. Chỉ đọc, không GPU.

## Điều tôi đo được, và điều tôi KHÔNG khẳng định

Sửa tolerance là **~1.00x** ở hình dạng thực tế (96 mode, 8 mẫu Fisher, `a != 0`).
Nó chỉ lên 3–10x khi `rho <= 1e-7` VÀ bước gần như trung tính bậc nhất với task
cũ (`1.a ~ 0`):

```
 |1.a|   |    rho=1e-4    1e-5     1e-6     1e-7     1e-8
1.4e-06  |     1.00x     1.00x    1.00x    1.00x    1.00x     <- chế độ thực tế
1.4e-07  |     1.00x     1.01x    1.21x    1.40x    1.44x
1.4e-08  |     1.00x     1.01x    1.41x    3.12x    6.93x
0        |     1.00x     1.01x    1.41x    3.32x   10.05x
```

Tôi đã nói với bạn con số 100x. Con số đó lấy từ một bài toán 1 chiều với `a = 0`,
không phải từ hình dạng của run này. **Sửa tolerance không giải thích được gì
trong Table 2.** Giữ nó vì một run không thể lập luận được dưới hai ngân sách
khác nhau, không phải vì nó sẽ làm đổi số.

Tần suất W1/W2 trong run thật thì tôi **không biết**. Đó đúng là câu hỏi cần
audit trả lời, và nó cần `steps.jsonl` của bạn.

## Một phát hiện có sẵn trong repo

`runs/diagnose_scalar/gate_trials.jsonl` — run thật trên RTX 5050 của bạn:

```
 update     headroom      scale  bt     logged    audited
    360    2.000e-02          1   0   accepted   accepted
    365    7.395e-03          1   0   accepted   accepted
    366    2.342e-03        0.5   1   accepted   accepted
    367    1.171e-04     0.0156   6   accepted   accepted
    368    4.657e-05    0.00781   7   accepted   accepted
    369    8.596e-06          0   9   rejected   rejected
```

Audit khớp với nhãn đã ghi — đúng như mong đợi, vì nhánh `scalar` không thể mắc
lỗi no-op ngầm (`z` của nó không bao giờ bằng 0; chỉ scale của gate xuống 0, và
cái đó đã được ghi là `rejected`).

Nhưng bảng này nói hai điều khác:

1. **Headroom sụp theo cấp số nhân**: 2e-2 → 8.6e-6 trong mười update. Ngân sách
   không tái tạo đang được tiêu hết trong một phần nhỏ của một task. Nghĩa là
   nhánh drift chạy ở `rho ~ 1e-5..1e-6` suốt phần lớn thời gian sau task đầu.

2. **Reject ở update 369 là artifact của lưới**: `diagnostic.json` của chính bạn
   ghi `finding: smaller_nonzero_step_is_feasible` — mức `2^-9` hợp lệ, lưới chỉ
   dừng ở `2^-8`. Nhánh `scalar` bị từ chối bước ở nơi một bước nhỏ hơn hai lần
   là hợp lệ, trong khi `drift` có solver liên tục để co bước. **Đây là confound
   giữa hai nhánh, và nó không nằm ở allocator.**

Tôi không sửa điểm 2. Nó đổi quyết định của cả hai nhánh, và theo đúng nguyên tắc
trong `SCALAR_DIAGNOSTIC.md` của bạn — nếu đổi lưới tìm bước thì phải áp dụng cho
cả scalar và drift, output mới, ghi rõ protocol. Đó là quyết định của bạn, không
phải của tôi.

## Nhánh scalar đúng

`method="scalar"` hiện tại = bước Muon đầy đủ + gate. Nó không phải là hạn chế
`z = c*1` của cùng bài toán. Hạn chế đúng KHÔNG phải là giải lại với `C` đường chéo:

```
max_c  c * (1.b)   s.t.  -(1.a) c + .5 c^2 (1.C.1) <= rho,   lower <= c <= 1
```

`1.C.1` giữ mọi số hạng chéo giữa các mode, tốn đúng một matvec trên `scores` đã
lưu. `allocate_scalar` giải nó bằng chính `solve_budget`, nên hai nhánh dùng chung
solver, chung tolerance và — trong runner — chung gate. `drift_scalar` được
chứng minh lồng trong `drift` ở cùng ngân sách (`tests/test_audit.py`), nên nếu
`drift` không thắng `drift_scalar` thì phân bổ per-mode không đáng công.

## Chạy

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audit.py -q

# audit log đã có — vài giây, không cần GPU
.\.venv\Scripts\python.exe audit_steps.py --run runs\pilot\drift_s1993 --output runs\audit\drift
.\.venv\Scripts\python.exe audit_steps.py --run runs\pilot\drift_s1993 runs\pilot\scalar_s1993 --output runs\audit\pilot
```

Log cũ không có `z_inf` sẽ được audit từ loss memory đo được và các số hạng của
allocator. `undecidable` đếm những record không mang cả hai — một record absent `z`
KHÔNG được đọc thành `z = 0`.

Đọc `noop_fraction_of_gated` **trước** khi đọc bất kỳ bảng accuracy nào. Một run mà
phần lớn update có gate là no-op thì không phải đang chạy method đang được đo.

```powershell
# nhánh scalar đúng, sau khi audit xong
.\.venv\Scripts\python.exe run_pilot.py --config configs\cifar100.yaml --output runs\pilot_v3 --methods drift drift_scalar --seeds 1993 --task-limit 2
```

## Giới hạn

Máy tạo gói này không có GPU. Đã chạy: 22 test mới + 20 test cũ của repo trên CPU,
cộng run end-to-end synthetic cho `drift` và `drift_scalar`. Một test cũ
(`test_huggingface_clip_adapter_path_without_download`) fail ở đây do phiên bản
`transformers` mới hơn của bạn — nó cũng fail trên bản checkout sạch trước khi tôi
sửa gì, không liên quan tới thay đổi này.

Chưa chạy CLIP/CIFAR thật. Test logic không chứng nhận CUDA, throughput hay chất
lượng phương pháp.

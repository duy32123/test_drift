# Phương pháp v0: phân bổ biên độ phổ của bước Muon

Đây là định nghĩa của **implementation được giao**, chưa phải tuyên bố tính mới
hoặc hiệu quả đã được xác lập. Code chạy trên LoRA factors, không trên ma trận
backbone hiệu dụng. Không suy ra cận spectral norm cho thay đổi của BA.

## Bước cơ sở và biến tối ưu

Với mỗi factor θℓ, tạo bước cơ sở Dℓ từ momentum/Nesterov và NS5, gồm luôn
learning rate và shape scaling. Quy ước cập nhật là θℓ ← θℓ − Dℓ.

Thin SVD: Dℓ = Uℓ diag(sℓ) Vℓᵀ. Định nghĩa Eℓi = sℓi uℓi vℓiᵀ, rồi đặt

    dℓ(z) = Σᵢ zℓi Eℓi,       θℓ ← θℓ − dℓ(z).

Mặc định 0 ≤ zℓi ≤ 1. z=1 tái dựng bước NS5 đến sai số FP32. Factor có bước
đúng bằng zero không tạo mode. Không giả định NS5 có singular value bằng 1.

Nối hệ số của tất cả factor thành z. Gradient task mới được chụp trước các
backward task cũ. Tọa độ:

    bℓi = ⟨∇θℓ Lnew, Eℓi⟩,
    aℓi = ⟨∇θℓ Lmemory, Eℓi⟩.

Do quy ước dấu trên, predicted new-loss reduction là bᵀz và predicted old-loss
change ở bậc nhất là −aᵀz. b không bị clamp; gradient mới có thể khác momentum.

## Độ cong và ngân sách

Mỗi hàng J lấy một ảnh memory, lấy một nhãn từ phân phối hiện tại của model,
tính gradient negative log likelihood của riêng ảnh đó rồi chiếu lên **tất cả**
Eℓi. Không dùng gradient trung bình batch làm score. Đặt

    C = JᵀJ/N + diagonal damping.

Damping của từng factor bằng λ nhân mean squared score của factor đó, với floor
1e−16 trước khi nhân λ. J giữ cross-factor terms; code dùng matvec để tránh ma
trận C đặc. Fisher là proxy PSD, không đồng nhất với Hessian loss có nhãn.

Mỗi ảnh memory có anchor là loss khi ảnh lần đầu được thêm vào bộ nhớ. Anchor
không cập nhật sau mỗi bước. Gọi H là mean anchor của memory đang giữ cộng δ.
Tại mỗi update, giải:

    maximize    bᵀz
    subject to  −aᵀz + ½ zᵀCz ≤ H − Lmemory(θ)
                z_lower ≤ z ≤ 1.

Một dual multiplier dùng chung cho mọi factor; inner solver dùng box L-BFGS-B.
Negative remaining budget được phép để yêu cầu recovery. Finite solver chỉ trả
candidate và diagnostic, không chứng nhận infeasibility toán học.

## Kiểm bước thật

Apply đồng thời các factor từ snapshot rồi đo lại mean loss trên toàn memory.
Nếu cần, thử fractions 1, 1/2, … theo cấu hình. Từ trạng thái đã khả thi, chỉ
nhận bước nếu measured loss ≤ H + tolerance. Nếu trạng thái ban đầu đã vượt H,
có thể nhận bước làm giảm loss; bước đó được đánh dấu `recovery`, không giả là
đã thỏa ngân sách. Reject khôi phục nguyên bit trọng số; momentum vẫn cập nhật
một lần theo minibatch mới, kể cả khi parameter step bị reject.

Memory loss dùng fixed all-class head và transform xác định. Loss mới và CIL
evaluation dùng tất cả lớp đã thấy. Xem PROTOCOL.md về giả định biết tên lớp.

## Đối chứng bắt buộc

- `muon`: áp dụng trực tiếp cùng bước cơ sở, không đọc memory.
- `scalar`: cùng memory, anchors và measured-loss gate; chỉ co toàn bộ bước bằng
  một scalar qua backtracking, không giải bài toán Fisher allocation.
- `drift_scalar`: đối chứng trực tiếp của per-mode allocation, giới hạn `z = c*1`.
  Với `b_s = 1^T b`, `a_s = 1^T a`, `C_s = 1^T C 1`, giải
  `max b_s*c` với `-a_s*c + 0.5*C_s*c^2 <= rho + tolerance` và `z_lower <= c <= 1`.
  `C_s` giữ toàn bộ cross-mode terms. Hai nhánh dùng cùng `solve_budget` và
  cùng `gate_tolerance` (mặc định `1e-6`) cho solver feasibility và measured-loss gate.
  Đây là đồng bộ tolerance; predicted loss vẫn là proxy của measured loss.
  So sánh ghép cặp dùng chung checkpoint task A qua `run_paired.py`.
- `replay`: trộn gradient dữ liệu cũ với gradient mới trước cùng Muon.
- `adamw`: baseline optimizer trên cùng các factor.

Hai loại chi phí khác nhau phải báo riêng: capacity memory và số lần đọc ảnh cũ.
v0 đọc toàn memory để lấy a và kiểm loss, cộng các Fisher backward. Cùng capacity
không có nghĩa là compute matched. Chưa cache Fisher qua các basis khác nhau.

Ngân sách memory không bảo đảm held-out forgetting, accuracy, hay từng task.
Mục tiêu thực nghiệm vẫn là frontier accuracy/forgetting sau tuning độc lập,
cùng báo wall-clock và peak VRAM; không dùng predicted drift thay kết quả đó.

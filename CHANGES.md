# Những thay đổi so với allocator đính kèm

Bản gốc được giữ nguyên trong `reference/`. Đây là một bộ chạy mới, không ghi đè
ba file đã tải lên và không dùng lời tuyên bố "18 test xanh" làm bằng chứng GPU.

1. **Đơn vị bước:** b, a, score columns chứa luôn LR, shape scaling và amplitudes
   NS5. Ngân sách áp dụng cho bước thật, không nhân eta lần nữa sau khi giải.
2. **Backend công bằng:** method SVD-decompose chính base update NS5. z=ones
   tái dựng nó đến sai số FP32; không thay NS5 bằng exact polar mà không gắn nhãn.
3. **Cross-factor terms:** nối score coordinates của mọi factor thành mỗi hàng J;
   C z = J.T@(J@z)/N + damping*z. Một dual multiplier toàn cục thay split budget.
4. **CPU/GPU rõ ràng:** SVD và backward trên device; QP nhỏ ở CPU float64. Tensor
   không vô tình được tạo trên CPU rồi cộng vào CUDA tensor.
5. **Zero update:** gradient/momentum bằng 0 tạo đúng update 0. Không ép rank ít
   nhất bằng 1 và di chuyển theo một vector SVD tùy ý.
6. **Rollback:** snapshot-copy thay phép cộng ngược, giữ nguyên bit FP32 khi reject.
7. **New gradient:** lưu trước khi đo memory; old signals dùng autograd.grad nên
   không thay p.grad của minibatch mới.
8. **Signed coefficients:** không clamp b. Với a>0, z<0 làm hại loss cũ ở bậc nhất;
   negative z giúp cả hai task khi cả a và b âm, không phải a dương/b âm.
9. **Feasibility:** solver báo residual và `infeasible_candidate`; finite iterations
   không được mô tả là chứng minh bất khả thi. Measured-loss gate quyết định áp dụng.
10. **CIL anchor:** per-example anchors bất biến trên fixed head; không reset ngân
    sách sau mỗi step hoặc lấy nhầm softmax mới làm loss reference cũ.
11. **Reproducibility:** cập nhật momentum đúng một lần/effective batch; checkpoint
    chứa RNG, anchors, adapter, momentum và optimizer; resume task-boundary có test.

Thiết kế hiện tại tối đa hóa alignment với gradient mới trong họ bước co giãn
các singular amplitudes của NS5. Khi b có thành phần âm, nghiệm hộp không ràng
buộc có thể khác z=ones. Vì vậy v0 không tuyên bố "budget lỏng luôn bằng Muon"
trừ khi các b_i đều không âm. Log ghi `negative_b_modes` để nhận diện trường hợp đó.

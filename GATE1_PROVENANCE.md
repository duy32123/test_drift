# Gate 1: source hash, audit và giới hạn chứng cứ

Hai manifest cũ ghi raw module SHA-256:
`db22df15dd0898d441f2366df68d16762227d94c81718ff41261c0618402d6d9`.
Source working tree trước sửa cơ chế fork cho đúng hash này. Chuẩn hóa CRLF thành LF
cho hash `d7de97f85014ba126f85b915fc914eefc126793f375ce35c6538b7bbda9dd8ea`,
trùng phép nối các Git blob `drift_cil/*.py` theo thứ tự tên ở commit `698fb93`.
Chênh lệch hash là do line endings. Không sửa các hash trong manifest lịch sử.

Tái chạy `audit_run` trên hai `steps.jsonl` cục bộ cho kết quả JSON bằng đúng audit
đã commit: scalar có 149 accepted / 211 rejected; drift có 354 accepted / 6 rejected.
Mỗi run có 360 update task A không gate và 360 update task B; không phát hiện no-op.
Log được đưa lên cùng bản cập nhật này để độc giả có thể tự kiểm:

```powershell
.\.venv\Scripts\python.exe audit_steps.py --run runs/gate1_corrected/drift_scalar_s1993 runs/gate1_corrected/drift_s1993
```

Sự phù hợp hash/log không phải bằng chứng độc lập rằng training đã dùng đúng source
hay rằng hai nhánh xuất phát từ cùng trạng thái. Task-A accuracy 97.6% và 97.8%
cho thấy đây chưa phải so sánh từ checkpoint chung. Checkpoint cũ vẫn lưu cục bộ;
không thể tạo ngược checkpoint task A chung cho hai run này.

Để khóa bằng chứng mới, chạy `run_paired.py` từ commit có cơ chế fork (hậu duệ
của `698fb93`), vào thư mục mới. Lưu source snapshot, checkpoint task A, checkpoint
hai nhánh, logs, audits, manifests và `checksums.json` thành artifact có thể tải.
Chưa có kết quả CIFAR ghép cặp mới trong bản cập nhật này. Không dùng smoke synthetic
làm bằng chứng accuracy cho paper.

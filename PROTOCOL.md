# Protocol và phạm vi kết luận

## Nguồn đối chiếu

- When Muon Meets Task Interference, Appendix D, Table 7:
  https://arxiv.org/html/2608.27518v1#A4
- Muon reference implementation:
  https://github.com/KellerJordan/Muon
- CLIP API:
  https://huggingface.co/docs/transformers/model_doc/clip
- MoE upstream (chỉ tham khảo; bộ chạy này không chứa router/DDAS):
  https://github.com/JiazuoYu/MoE-Adapters4CL

Appendix D xác nhận CLIP ViT-B/16 frozen, plain LoRA trên attention + MLP,
CIFAR-100 chia 10 task × 10 lớp, và NS5 cho Muon. Nó không đủ để tái dựng mọi
chi tiết của LoRA4CIL. Các số LR, batch 128, warmup 500 xuất hiện rõ trong phần
model merging, có câu nói dùng cấu hình Muon xuyên paper; điều đó vẫn không
giải quyết thiếu class order, LoRA rank/alpha, prompts, augmentation và training
schedule chi tiết của CL. Bộ chạy ghi rõ các lựa chọn thay vì đoán là đã khớp.

## Cấu hình v0: những lựa chọn cần đối chiếu thêm

| Mục | Cài đặt |
|---|---|
| Backbone | Hugging Face OpenAI CLIP ViT-B/16, revision được ghi trong manifest |
| Trainable | Một bộ LoRA dùng chung cho cả chuỗi; q/k/v/out và fc1/fc2 mỗi block |
| Rank/alpha | 16/16, **lựa chọn v0**, chưa xác minh với code tác giả |
| Bias/LayerNorm/router | Không train; plain LoRA không có router; không có optimizer phụ |
| Text head | Một prompt `a photo of a {}.`, frozen; chưa đối chiếu prompt ensemble của paper |
| Main training | Cross-entropy trên tất cả lớp đã thấy; không chỉ lớp của task hiện tại |
| CIL evaluation | Argmax trên toàn bộ lớp đã thấy, không dùng task-ID |
| Memory objective | Fixed 100-class head, giữ nguyên mẫu số khi lớp đã thấy tăng |
| Future information | Chỉ tên 100 lớp từ đầu; không dùng ảnh/nhãn huấn luyện của task tương lai |
| Memory | Tổng capacity 200 ảnh, class-balanced, training-only, immutable per-example anchors |
| Fisher | Model-sampled label, mỗi score từ một ảnh; 8 score/update là mặc định phát triển |
| Schedule | Cosine **toàn chuỗi**, warmup 500 update; 10 epoch/task là lựa chọn v0 |
| Data | 450 train + 50 validation/class; test giữ ngoài tuning |
| NS | Fixed quintic, 5 vòng, FP32, Frobenius input norm, shape factor sqrt(max(1,m/n)) |
| AdamW arm | Cùng factor trainable, zero weight decay; tune LR riêng |
| Statistical replication | Seed model/optimizer tách khỏi seed class order; ghép cặp theo seed |

Việc dùng fixed 100-class memory head là một lựa chọn phương pháp rõ ràng: nó
tránh reset ngân sách do softmax mở rộng. Nếu benchmark cấm biết tên các lớp
tương lai, phải thay protocol này trước khi sử dụng. Không được âm thầm đổi sang
task-specific head rồi gọi đó là bảo vệ CIL.

Budget gộp toàn bộ memory có thể cho phép lợi ích task này bù thiệt hại task khác.
Nó không phải ràng buộc riêng cho từng task, cũng không chứng minh accuracy
held-out được bảo vệ. Luôn xem per-task accuracy và BWT.

## Quy tắc so sánh

1. Reproduce AdamW/Muon trong cùng harness trước. Số khác Table 7 không tự nó là
   bằng chứng bug hay method thắng; phải kiểm các khác biệt trong bảng trên.
2. Scalar và drift phải dùng cùng base backend, LR/shape scaling, memory samples,
   head tham chiếu và acceptance rule. Chỉ drift có phân bổ riêng từng spectral mode.
3. Đối chiếu NS5 với polar phải là ablation được gắn nhãn, không trộn hai backend.
4. Memory access là tài nguyên. Muon/replay/scalar/drift đọc số ảnh cũ khác nhau;
   báo cáo cả image reads và thời gian, không gọi "compute matched" chỉ vì cùng steps.
5. Các statistic C, a, b dùng current basis. v0 chưa cache Fisher qua nhiều step;
   thêm refresh interval đòi hỏi xử lý đổi cơ sở và một ablation riêng.
6. Không suy ra tính mới từ kết quả pilot. Cần đối chiếu GEM/A-GEM, Fisher/replay
   và các phương pháp spectral CL trước khi định vị đóng góp.

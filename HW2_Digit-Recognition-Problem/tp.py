import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


import torch
import torchvision.ops

if torch.cuda.is_available():
    try:
        boxes = torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150]], dtype=torch.float32).cuda()
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()
        iou_threshold = 0.5
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        print("NMS ran successfully on CUDA. Keep indices:", keep_indices)
    except Exception as e:
        print(f"Error running NMS on CUDA: {e}")
else:
    print("CUDA is not available.")
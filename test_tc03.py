import cv2, sys, os
sys.path.append(r'd:\tam\stu\xulianhso\ORM_demo1 (1)')
from services.grading_service.omr_service import OMRService

tc_dir = r'd:\tam\stu\xulianhso\ORM_demo1 (1)\resources\test image xu ly anh cuoi ky\test image xu ly anh cuoi ky\TC03_thieu_sang'
files = sorted([f for f in os.listdir(tc_dir) if f.endswith('.jpg')])

service = OMRService()
for f in files:
    image = cv2.imread(os.path.join(tc_dir, f))
    scan = service.pre_scan_image(image)
    sbd = scan.get('sbd', '?')
    made = scan.get('made', '?')
    err = scan.get('error', '')
    short = f[:30]
    print(f"{short}  SBD={sbd}  MADE={made}  err={err[:50] if err else ''}")

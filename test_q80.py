import cv2, sys, os
sys.path.append(r'd:\tam\stu\xulianhso\ORM_demo1 (1)')
from services.grading_service.omr_service import OMRService

tc_dir = r'd:\tam\stu\xulianhso\ORM_demo1 (1)\resources\test image xu ly anh cuoi ky\test image xu ly anh cuoi ky\TC03_thieu_sang'
f = 'z7965998933269_28de2f9b60b1d6e70e401d36c0ffce1c.jpg'
image = cv2.imread(os.path.join(tc_dir, f))

service = OMRService()
out_dir = r'C:\Users\nguye\.gemini\antigravity-ide\brain\ba3822d5-3a12-4e1e-980f-cd06a7d03d9a\scratch'
os.makedirs(out_dir, exist_ok=True)

answers = {}
for i in range(1, 121):
    answers[str(i)] = 'D'  # Set all to D so we can see what was detected

graded, report, score = service.grade_image(image, answers=answers, debug_dir=out_dir, debug_prefix='tc03_fix_')
print("SBD:", score.get('sbd'))
print("MADE:", score.get('made'))

# Print questions around 80
for q in range(78, 83):
    key = str(q)
    detected = report.get(key, {})
    print(f"Q{q}: {detected}")

cv2.imwrite(os.path.join(out_dir, 'graded_fixed.jpg'), graded)
print("Saved graded image")

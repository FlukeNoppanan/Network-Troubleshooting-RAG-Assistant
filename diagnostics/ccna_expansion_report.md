# รายงานขยายคลัง CCNA พื้นฐาน

ขยายจาก 18 เป็น 48 เอกสาร เพิ่ม 30 เอกสารที่เขียนใหม่เป็นภาษาไทยพร้อมศัพท์อังกฤษ ตัวอย่างคำสั่ง lab และแนวทาง troubleshooting ครอบคลุมหกหมวด CCNA v1.1 ดู [ตารางหัวข้อ](../CCNA_COVERAGE.md)

เพิ่มเติม FTP/TFTP และการสำรอง configuration ในเอกสาร 17 พร้อม local username/enable secret ในเอกสาร 34 เก็บเอกสารเดิมทั้งหมด

## ขนาดคลัง

- 48 เอกสาร; 87,131 ตัวอักษรก่อน clean, 87,082 หลัง clean
- 261 chunks และ FAISS vectors, 384 มิติ
- สูงสุด 128 tokens ต่อ chunk; normalized document/query vectors
- Threshold เดิม 0.38; GROUNDED top score ต่ำสุด 0.4304; NOT_FOUND สูงสุด 0.3269

## การตรวจสอบ

- `HF_HUB_OFFLINE=1 .venv/bin/python test_rag.py`: 8 tests ผ่าน (18.400 วินาที)
- `HF_HUB_OFFLINE=1 .venv/bin/python debug_retrieval.py`: 59/59 กรณีผ่าน รวม 4 NOT_FOUND
- คำถาม OSPF EXSTART ยังได้เอกสาร 05 เป็นอันดับแรก คะแนน 0.6192
- [ผลรายคำถาม](retrieval_results.csv), [รายละเอียดและสถิติ](retrieval_details.json)

ปรับรูปแบบเอกสารใหม่ให้ชื่อเรื่องอยู่ติดกับย่อหน้าอธิบาย เพื่อป้องกัน chunk ที่มีเพียงชื่อเรื่องสั้นและจับคู่คำถามนอกคลังได้ง่าย ไม่มีการแก้ retrieval, FAISS, embedding, chunker, threshold, UI, Groq integration หรือ NOT_FOUND logic การทดสอบเรียก embedding ในเครื่องและจำลอง Groq เท่านั้น ไม่มี real API request

เพิ่ม regression 32 กรณีสำหรับเอกสารใหม่และเนื้อหาบริการเพิ่มเติม พร้อมเพิ่มเกณฑ์จำนวนข้อมูลในชุดทดสอบเดิม การตรวจ retrieval ยืนยันแหล่งข้อมูลและ threshold ไม่ใช่การประเมินคุณภาพคำตอบจาก LLM หรือการรับรองว่าครอบคลุมทุกข้อสอบ

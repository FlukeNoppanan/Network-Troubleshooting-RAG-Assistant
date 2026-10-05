# รายงานการแก้ retrieval (5 ตุลาคม 2026)

## สิ่งที่พิสูจน์ได้และสาเหตุ

ก่อนแก้ คำถาม “OSPF Neighbor ค้างอยู่ที่ EXSTART เกิดจากอะไร?” ได้ 05_ospf.txt อันดับ 1 คะแนน 0.6438 และ chunk ที่กล่าวถึง MTU อันดับ 2 คะแนน 0.5307 ดังนั้นคำถามนี้ไม่ได้ถูกตัดโดย threshold 0.25 หรือ 0.38 ในการทดสอบโดยตรงครั้งนี้ ไม่พบข้อผิดพลาดด้าน cosine normalization, score direction หรือ index-to-chunk mapping และไม่สามารถยืนยันสาเหตุของคำตอบเก่าใน session ผู้ใช้โดยไม่มี response เดิมของ Gemini

พบข้อบกพร่องจริง 3 จุด:

1. normalize ทั้งข้อความทำให้ blank paragraphs หาย และ sliding overlap เริ่มกลางคำ ตัวอย่าง chunk OSPF เริ่ม “ate เพิ่ม” หรือ “ยวข้องบน segment”
2. OSPF chunks เดิมมี 220/162/161/119 tokens แต่โมเดลใช้ได้ 128 tokens ทำให้ส่วนท้ายไม่ถูกใช้ในการ embed ปรับให้ normalize รายบรรทัด รักษาย่อหน้า และตรวจ token budget จริง มี overlap เฉพาะเมื่อใส่แล้วไม่เกิน budget
3. validate_answer เดิมแปลง JSON/citation ที่ผิดเป็น NOT_FOUND และปฏิเสธแม้ structured citations ถูกต้องหากไม่มี inline IDs ทำให้ผู้ใช้ไม่เห็นความแตกต่างระหว่างไม่มีข้อมูลกับ format error แก้ให้เติม inline IDs จาก structured citations ที่ตรวจแล้ว และแสดง error แยกสำหรับ JSON/citation ที่ไม่ถูกต้อง ไม่อนุญาต ID ที่ไม่ได้ retrieve

เพิ่ม OSPF master/slave และ DBD sequence negotiation พร้อมแยก Hello/Dead, area และ authentication mismatch จากสาเหตุ EXSTART เพื่อไม่ทำให้คำอธิบายวินิจฉัยผิด ไม่มี LLM classifier เพิ่ม ไม่มี lexical boost หรือคำถามที่ hard-code ให้ตอบ/ปฏิเสธ

## OSPF Top-5 ก่อน/หลัง

| Rank | Before source/chunk | Before cosine | After source/chunk | After cosine |
|---|---|---:|---|---:|
| 1 | 05_ospf.txt / 1 | 0.6438 | 05_ospf.txt / 1 | 0.6192 |
| 2 | 05_ospf.txt / 3 | 0.5307 | 05_ospf.txt / 4 | 0.5650 |
| 3 | 05_ospf.txt / 2 | 0.4125 | 05_ospf.txt / 5 | 0.5037 |
| 4 | 01_osi_tcpip.txt / 1 | 0.3864 | 05_ospf.txt / 6 | 0.4823 |
| 5 | 08_firewall_acl.txt / 3 | 0.3824 | 05_ospf.txt / 2 | 0.4342 |

คะแนนอันดับ 1 หลังแก้ไม่จำเป็นต้องเพิ่ม เป้าหมายคือหลักฐานเฉพาะครบโดยไม่ถูก truncate หลังแก้ทั้ง 5 ผลเป็น OSPF; chunk 4 อธิบาย EXSTART/MTU/master-slave และ chunk 5 เป็นวิธีตรวจสอบ แสดง text preview ได้ด้วย `python debug_retrieval.py` และข้อความเต็มอยู่ใน JSON รายละเอียด

## Calibration ทุกคำถาม

PASS ของ GROUNDED หมายถึง expected source อยู่ใน Top-5 และผ่าน threshold ไม่ได้หมายถึงตรวจคำตอบ Gemini ของทุกคำถาม คำถาม subnet/default gateway มี expected source อันดับ 2; คำถาม GROUNDED อื่นมี expected source อันดับ 1

| ID | Question | Expected type | Expected source | Top source | Top score | Result |
|---|---|---|---|---|---:|---|
| 1 | OSPF neighbor stuck in EXSTART or EXCHANGE what should I check? | GROUNDED | 05_ospf.txt | 05_ospf.txt | 0.5787 | PASS |
| 2 | เครื่องต่าง VLAN ติดต่อกันไม่ได้ต้องตรวจอะไร? | GROUNDED | 03_vlan_trunking.txt | 03_vlan_trunking.txt | 0.7556 | PASS |
| 3 | DHCP client cannot obtain an IP address how do I troubleshoot? | GROUNDED | 06_dhcp_dns.txt | 06_dhcp_dns.txt | 0.8108 | PASS |
| 4 | DNS resolution fails but access by IP works what should I check? | GROUNDED | 06_dhcp_dns.txt | 06_dhcp_dns.txt | 0.7763 | PASS |
| 5 | How do I troubleshoot outbound NAT translations? | GROUNDED | 07_nat.txt | 07_nat.txt | 0.6436 | PASS |
| 6 | เกิด STP loop และ MAC flapping ต้องตรวจอะไร? | GROUNDED | 04_stp_rstp.txt | 04_stp_rstp.txt | 0.6035 | PASS |
| 7 | VPN ส่ง packet เล็กได้แต่เว็บค้าง ต้องตรวจ MTU และ MSS อย่างไร? | GROUNDED | 09_vpn.txt | 09_vpn.txt | 0.7933 | PASS |
| 8 | How do I troubleshoot a firewall rule blocking TCP traffic? | GROUNDED | 08_firewall_acl.txt | 08_firewall_acl.txt | 0.7785 | PASS |
| 9 | ตั้ง subnet mask หรือ default gateway ผิดมีอาการอย่างไร? | GROUNDED | 02_ip_subnetting.txt | 03_vlan_trunking.txt | 0.669 | PASS |
| 10 | How do I troubleshoot TCP retransmission and duplicate ACK? | GROUNDED | 10_network_troubleshooting.txt | 10_network_troubleshooting.txt | 0.8072 | PASS |
| 11 | How do I deploy Kubernetes with Helm? | NOT_FOUND | — | 04_stp_rstp.txt | 0.2883 | PASS |
| 12 | How do I optimize a PostgreSQL query? | NOT_FOUND | — | 10_network_troubleshooting.txt | 0.2802 | PASS |
| 13 | ARP ใช้ทำอะไรเมื่อปลายทางอยู่นอก subnet? | GROUNDED | 01_osi_tcpip.txt | 01_osi_tcpip.txt | 0.7205 | PASS |
| 14 | OSPF DROTHER neighbors stay in 2-Way is that normal? | GROUNDED | 05_ospf.txt | 05_ospf.txt | 0.4659 | PASS |
| 15 | OSPF Neighbor ค้างอยู่ที่ EXSTART เกิดจากอะไร? | GROUNDED | 05_ospf.txt | 05_ospf.txt | 0.6192 | PASS |
| 16 | Client ไม่ได้รับ IP Address จาก DHCP ควรตรวจสอบอะไร? | GROUNDED | 06_dhcp_dns.txt | 06_dhcp_dns.txt | 0.8007 | PASS |
| 17 | VLAN เดียวกันแต่เครื่องสองเครื่องสื่อสารกันไม่ได้ ควรตรวจสอบอะไร? | GROUNDED | 03_vlan_trunking.txt | 03_vlan_trunking.txt | 0.7837 | PASS |
| 18 | วิธี Deploy Kubernetes ด้วย Helm ทำยังไง? | NOT_FOUND | — | 04_stp_rstp.txt | 0.2812 | PASS |

เลือก threshold 0.38 จากช่องว่างระหว่างค่าสูงสุดของ NOT_FOUND 0.2883 และค่าต่ำสุดของ GROUNDED 0.4659 (กึ่งกลางประมาณ 0.3771) ใช้เกณฑ์ `score >= threshold` ต่อ chunk ไม่มีเหตุผลจากชุดนี้ที่จะเพิ่ม strategy หลายสัญญาณ คะแนนเป็น cosine similarity ไม่ใช่ probability ชุดทดสอบขนาดนี้ไม่รับประกันคำถามนอกคลังทุกแบบ

Kubernetes ภาษาไทย/อังกฤษและ PostgreSQL ถูกปฏิเสธตรงข้อความ “ไม่พบข้อมูลในคลังเอกสาร” โดยไม่เรียก Gemini (ทดสอบด้วย mock ที่ assert ว่า API ไม่ถูกเรียก)

## ผลตรวจระบบ

- documents: 10
- raw_characters: 22567
- cleaned_characters: 22557
- chunks: 63
- vectors: 63
- dimensions: 384
- max_chunk_tokens: 128
- model_token_limit: 128
- query_norm: 1.0000001192092896
- document_norm_min: 0.9999999403953552
- document_norm_max: 1.0000001192092896
- min_grounded_top_score: 0.4659
- max_not_found_top_score: 0.2883
- threshold: 0.38

- automated tests: 7 ผ่าน รวม 18 retrieval cases, token budget, paragraph preservation, normalization และเทียบ FAISS กับ dot product ตรง ๆ, source mapping, citation validation, generation mock และ Streamlit AppTest รวม OSPF grounded answer พร้อม source expander และ Retrieval Debug
- Streamlit AppTest: ไม่มี exception; missing-secret warning, session history, NOT_FOUND และ Clear Chat ทำงาน ใช้ placeholder secret เฉพาะใน test จึงไม่เรียก API จริง
- pip check และ compile: ผ่าน
- Streamlit ที่เปิดใหม่: http://127.0.0.1:8503; health endpoint ตอบ ok ไม่มี startup error พอร์ต 8502 มี process ใช้งานอยู่จึงเลือก 8503
- Gemini integration: ใช้ gemini-3.1-flash-lite เดิม คำถาม OSPF จริงสำเร็จพร้อม 05_ospf.txt chunk 4 หลัง retry จาก HTTP 503 ครั้งแรก ผลอยู่ใน gemini_integration.json
- ไม่มีการแก้ .streamlit/secrets.toml, ไม่พิมพ์ key, ไม่ commit/push

## ไฟล์เปลี่ยนแปลง

app.py, data/05_ospf.txt, test_rag.py, test_questions.csv, README.md; เพิ่ม debug_retrieval.py และ diagnostics/ (รายงาน, ตาราง CSV, JSON ก่อน/หลังและ integration)

## สิ่งที่ยังไม่ยืนยัน

ไม่ได้ทำ visual inspection ด้วย browser และยังไม่ได้เรียก Gemini จริงทุก test case เพื่อประหยัด quota ไม่สามารถระบุได้ว่าคำตอบ NOT_FOUND เก่าเกิดจาก Gemini ปฏิเสธเองหรือจาก validator เพราะไม่มี response เก่าที่เก็บไว้ ตอนนี้ Retrieval Debug แยก stage เหล่านี้ได้ หากยังเห็นปัญหาให้ดู status แทนการลด threshold ข้อจำกัดของ embedding และ Gemini ยังทำให้ต้องประเมินคำถามใหม่เมื่อคลังเปลี่ยน
